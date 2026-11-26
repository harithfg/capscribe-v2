"""
CapScribe V2 - Web-Based Accessible Lecture Recording
Backend: FastAPI + Deepgram Nova-3 (diarization + keyterms + code-switching)
         + Groq Whisper (fallback STT) + Groq LLM (study summary)
         + Auto-detect Dosen (pembicara dominan)
"""
# Tambahkan di import bagian atas app.py:
import struct
import zlib
from functools import lru_cache
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
import io
import os
import re
import secrets
from datetime import datetime
from typing import Optional

import httpx
from docx import Document
from docx.shared import Pt, RGBColor
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Konfigurasi Environment
# ---------------------------------------------------------------------------
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
CAPSCRIBE_USER = os.environ.get("CAPSCRIBE_USER", "").strip()
CAPSCRIBE_PASS = os.environ.get("CAPSCRIBE_PASS", "").strip()

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_SUMMARY_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
PARAGRAPH_GAP_S = 2.5
MAX_BLOCK_CHARS = 420

CORRECTIONS = {
    "mikro kontroler": "mikrokontroler",
    "ardu ino": "Arduino",
}

HALLUCINATION_PATTERNS = [
    r"(?i)terima kasih telah menonton\.?",
    r"(?i)terima kasih sudah menonton\.?",
    r"(?i)jangan lupa (like|subscribe|komen).*?$",
    r"(?i)subtitles? by .*?$",
    r"(?i)thanks for watching\.?",
    r"(.)\1{6,}",
]

# ---------------------------------------------------------------------------
# App & Optional Basic Auth
# ---------------------------------------------------------------------------
app = FastAPI(title="CapScribe V2", version="2.1.0")
security = HTTPBasic(auto_error=False)


def verify_auth(credentials: Optional[HTTPBasicCredentials] = Depends(security)):
    if not (CAPSCRIBE_USER and CAPSCRIBE_PASS):
        return True
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login diperlukan",
            headers={"WWW-Authenticate": "Basic"},
        )
    ok_user = secrets.compare_digest(credentials.username, CAPSCRIBE_USER)
    ok_pass = secrets.compare_digest(credentials.password, CAPSCRIBE_PASS)
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username atau password salah",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Utilitas Teks & Diarization
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.strip()
    for pat in HALLUCINATION_PATTERNS:
        text = re.sub(pat, "", text).strip()
    for wrong, right in CORRECTIONS.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def parse_keyterms(terms_str: str) -> list[str]:
    if not terms_str:
        return []
    raw = re.split(r"[,;\n]+", terms_str)
    seen, out = set(), []
    for item in raw:
        t = item.strip()
        if t and len(t) <= 60 and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
        if len(out) >= 95:
            break
    return out


def smooth_word_speakers(words: list[dict]) -> list[dict]:
    """Hilangkan kedipan pembicara singkat (<0.35s) di tengah kalimat pembicara yang sama."""
    if len(words) < 3:
        return words
    for i in range(1, len(words) - 1):
        prev_s = words[i - 1].get("speaker")
        curr_s = words[i].get("speaker")
        next_s = words[i + 1].get("speaker")
        dur = float(words[i].get("end", 0)) - float(words[i].get("start", 0))
        if prev_s is not None and prev_s == next_s and curr_s != prev_s and dur < 0.35:
            words[i]["speaker"] = prev_s
    return words


def detect_primary_speaker(blocks: list[dict]) -> Optional[int]:
    """Pembicara dengan total kata terbanyak dianggap sebagai Dosen (Utama)."""
    counts: dict[int, int] = {}
    for b in blocks:
        spk = b.get("speaker")
        if spk is not None:
            counts[spk] = counts.get(spk, 0) + len(b.get("text", "").split())
    if not counts:
        return None
    return max(counts, key=counts.get)


def build_blocks_from_deepgram(js: dict, use_diarize: bool) -> tuple[list[dict], float]:
    meta = js.get("metadata", {})
    duration_s = float(meta.get("duration") or 0.0)

    channels = js.get("results", {}).get("channels", [])
    if not channels:
        return [], duration_s
    alt = channels[0].get("alternatives", [{}])[0]
    words = alt.get("words", [])

    if not words:
        raw = clean_text(alt.get("transcript", ""))
        if not raw:
            return [], duration_s
        return [{"speaker": 1 if use_diarize else None, "start": 0.0, "end": duration_s, "text": raw}], duration_s

    if use_diarize:
        words = smooth_word_speakers(words)

    speaker_map = {}
    blocks = []

    for w in words:
        token = (w.get("punctuated_word") or w.get("word") or "").strip()
        if not token:
            continue
        start = float(w.get("start", 0.0))
        end = float(w.get("end", start))

        if use_diarize:
            raw_spk = w.get("speaker")
            raw_spk = 0 if raw_spk is None else int(raw_spk)
            spk = speaker_map.setdefault(raw_spk, len(speaker_map) + 1)
        else:
            spk = None

        last = blocks[-1] if blocks else None
        same_speaker = last is not None and last["speaker"] == spk
        small_gap = last is not None and (start - last["end"]) <= PARAGRAPH_GAP_S
        sentence_done = last is not None and last["text"].endswith((".", "!", "?"))
        too_long = last is not None and len(last["text"]) >= MAX_BLOCK_CHARS and sentence_done

        if same_speaker and small_gap and not too_long:
            last["text"] += " " + token
            last["end"] = end
        else:
            blocks.append({"speaker": spk, "start": start, "end": end, "text": token})

    cleaned_blocks = []
    for b in blocks:
        txt = clean_text(b["text"])
        if txt:
            b["text"] = txt
            cleaned_blocks.append(b)

    return cleaned_blocks, duration_s


def build_blocks_from_groq(js: dict) -> tuple[list[dict], float]:
    segments = js.get("segments", [])
    duration_s = float(js.get("duration") or 0.0)
    blocks = []
    for s in segments:
        txt = clean_text(s.get("text", ""))
        if not txt:
            continue
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start))
        last = blocks[-1] if blocks else None
        if last and (start - last["end"]) <= PARAGRAPH_GAP_S and len(last["text"]) < MAX_BLOCK_CHARS:
            last["text"] += " " + txt
            last["end"] = end
        else:
            blocks.append({"speaker": None, "start": start, "end": end, "text": txt})
    if not blocks and js.get("text"):
        txt = clean_text(js["text"])
        if txt:
            blocks.append({"speaker": None, "start": 0.0, "end": duration_s, "text": txt})
    return blocks, duration_s


# ---------------------------------------------------------------------------
# Engine Pemanggil API
# ---------------------------------------------------------------------------
async def transcribe_deepgram(
    audio_bytes: bytes, content_type: str, language: str, terms: str, diarize: bool
) -> tuple[list[dict], float]:
    lang_code = {"id": "id", "en": "en", "multi": "multi"}.get(language, "multi")
    params: list[tuple[str, str]] = [
        ("model", "nova-3"),
        ("language", lang_code),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("utterances", "true"),
        ("diarize", "true" if diarize else "false"),
    ]
    for kt in parse_keyterms(terms):
        params.append(("keyterm", kt))

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": content_type or "application/octet-stream",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=20.0)) as client:
        resp = await client.post(DEEPGRAM_URL, params=params, headers=headers, content=audio_bytes)
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Deepgram error ({resp.status_code}): {resp.text[:300]}")
        js = resp.json()

    return build_blocks_from_deepgram(js, diarize)


async def transcribe_groq(
    audio_bytes: bytes, filename: str, content_type: str, language: str, terms: str
) -> tuple[list[dict], float]:
    prompt = "Berikut adalah transkrip kuliah dalam bahasa Indonesia atau Inggris."
    if terms.strip():
        prompt += f" Istilah: {terms.strip()}"

    data = {
        "model": "whisper-large-v3",
        "temperature": "0",
        "response_format": "verbose_json",
        "prompt": prompt[:500],
    }
    if language in ("id", "en"):
        data["language"] = language

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=20.0)) as client:
        resp = await client.post(
            GROQ_STT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (filename or "audio.webm", audio_bytes, content_type or "audio/webm")},
            data=data,
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Groq Whisper error ({resp.status_code}): {resp.text[:300]}")
        js = resp.json()

    return build_blocks_from_groq(js)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
# Tambahkan fungsi & route PWA ini di app.py (setelah @app.get("/")):
@lru_cache(maxsize=2)
def _generate_icon_png(size: int) -> bytes:
    """Buat ikon PNG biru + simbol mikrofon putih tanpa perlu file gambar eksternal."""
    rows = bytearray()
    cx, cy = size / 2.0, size / 2.0
    for y in range(size):
        rows.append(0)  # filter type 0
        t = y / size
        bg_r = int(30 + (37 - 30) * t)
        bg_g = int(58 + (99 - 58) * t)
        bg_b = int(138 + (235 - 138) * t)
        for x in range(size):
            nx = (x - cx) / size
            ny = (y - cy) / size
            # Kapsul mic + tiang + alas
            in_capsule = (abs(nx) <= 0.11 and -0.22 <= ny <= 0.04) or (
                nx * nx + (ny + 0.22) ** 2 <= 0.11 ** 2
            ) or (nx * nx + (ny - 0.04) ** 2 <= 0.11 ** 2)
            r_arc = (nx * nx + (ny - 0.04) ** 2) ** 0.5
            in_arc = ny >= 0.04 and (0.17 <= r_arc <= 0.215)
            in_stem = abs(nx) <= 0.025 and 0.20 <= ny <= 0.30
            in_base = abs(nx) <= 0.13 and 0.29 <= ny <= 0.335
            if in_capsule or in_arc or in_stem or in_base:
                rows.extend((255, 255, 255))
            else:
                rows.extend((bg_r, bg_g, bg_b))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack("!IIBBBBB", size, size, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(rows), 9)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


@app.get("/manifest.json")
async def pwa_manifest():
    return FileResponse("static/manifest.json", media_type="application/manifest+json")


@app.get("/sw.js")
async def pwa_sw():
    return FileResponse("static/sw.js", media_type="application/javascript")


@app.get("/icon-192.png")
async def pwa_icon_192():
    return Response(content=_generate_icon_png(192), media_type="image/png")


@app.get("/icon-512.png")
async def pwa_icon_512():
    return Response(content=_generate_icon_png(512), media_type="image/png")
async def index(_: bool = Depends(verify_auth)):
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "deepgram_ready": bool(DEEPGRAM_API_KEY),
        "groq_ready": bool(GROQ_API_KEY),
        "auth_enabled": bool(CAPSCRIBE_USER and CAPSCRIBE_PASS),
    }


@app.post("/api/transcribe")
async def api_transcribe(
    file: UploadFile = File(...),
    language: str = Form("id"),
    terms: str = Form(""),
    diarize: bool = Form(True),
    engine: str = Form("auto"),
    _: bool = Depends(verify_auth),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="File audio kosong. Rekam atau unggah audio terlebih dahulu.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Ukuran file melebihi batas 100 MB.")

    content_type = file.content_type or "audio/webm"
    use_deepgram = (engine in ("auto", "deepgram")) and bool(DEEPGRAM_API_KEY)

    notice = ""
    if use_deepgram:
        try:
            blocks, duration_s = await transcribe_deepgram(audio_bytes, content_type, language, terms, diarize)
            engine_used = "Deepgram Nova-3"
        except HTTPException as exc:
            if engine == "auto" and GROQ_API_KEY:
                blocks, duration_s = await transcribe_groq(
                    audio_bytes, file.filename or "audio.webm", content_type, language, terms
                )
                engine_used = "Groq Whisper (Cadangan)"
                notice = "Deepgram sedang bermasalah, otomatis beralih ke Groq (tanpa pemisahan pembicara)."
            else:
                raise exc
    elif GROQ_API_KEY:
        blocks, duration_s = await transcribe_groq(
            audio_bytes, file.filename or "audio.webm", content_type, language, terms
        )
        engine_used = "Groq Whisper"
        if diarize:
            notice = "Pemisahan pembicara membutuhkan DEEPGRAM_API_KEY (saat ini memakai Groq cadangan)."
    else:
        raise HTTPException(
            status_code=500,
            detail="Belum ada API key yang aktif. Isi DEEPGRAM_API_KEY (atau GROQ_API_KEY) di environment server.",
        )

    if not blocks:
        raise HTTPException(status_code=422, detail="Tidak ada ucapan yang terdeteksi dalam audio.")

    speakers = sorted({b["speaker"] for b in blocks if b.get("speaker") is not None})
    primary_speaker = detect_primary_speaker(blocks)

    return JSONResponse({
        "blocks": blocks,
        "speakers": speakers,
        "primary_speaker": primary_speaker,
        "duration_s": round(duration_s, 1),
        "engine_used": engine_used,
        "notice": notice,
    })


class SummarizeRequest(BaseModel):
    transcript: str
    terms: str = ""
    language: str = "id"
    primary_speaker_name: str = ""


@app.post("/api/summarize")
async def api_summarize(req: SummarizeRequest, _: bool = Depends(verify_auth)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Fitur ringkasan membutuhkan GROQ_API_KEY di server.")
    text = req.transcript.strip()
    if len(text) < 40:
        raise HTTPException(status_code=400, detail="Transkrip terlalu pendek untuk diringkas.")

    clipped = text[:28000]
    lang_hint = "Bahasa Indonesia" if req.language != "en" else "English"
    lecturer_note = (
        f" Jadikan ucapan dari '{req.primary_speaker_name}' sebagai sumber materi utama, "
        "dan abaikan obrolan sampingan pembicara lain kecuali berupa pertanyaan kuliah yang relevan."
        if req.primary_speaker_name.strip()
        else ""
    )
    system_prompt = (
        f"Kamu adalah asisten belajar mahasiswa yang teliti. Tulis output dalam {lang_hint} yang rapi dan jelas. "
        "Abaikan basa-basi, sapaan, atau obrolan di luar topik kuliah. Fokus pada konsep inti agar mahasiswa tidak kehilangan poin penting."
        + lecturer_note
    )
    user_prompt = (
        f"Istilah/topik kuliah: {req.terms or '-'}\n\n"
        "Buat ringkasan belajar terstruktur dari transkrip berikut dengan format persis 3 bagian ini:\n"
        "### 1. Poin Utama Kuliah\n(3-6 poin padat tentang materi inti)\n\n"
        "### 2. Istilah & Konsep Penting\n(Daftar istilah teknis dari kuliah beserta penjelasan singkat 1 kalimat)\n\n"
        "### 3. Pertanyaan / Catatan Penting\n(Poin diskusi/tanya jawab atau hal yang perlu dipelajari ulang; tulis 'Tidak ada catatan khusus' jika tidak ada)\n\n"
        f"TRANSKRIP:\n{clipped}"
    )

    last_error = ""
    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0)) as client:
        for model in GROQ_SUMMARY_MODELS:
            resp = await client.post(
                GROQ_CHAT_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1400,
                },
            )
            if resp.status_code < 400:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                return {"summary": content, "model": model}
            last_error = f"{model}: {resp.status_code} {resp.text[:200]}"

    raise HTTPException(status_code=502, detail=f"Gagal membuat ringkasan ({last_error})")


class ExportBlock(BaseModel):
    speaker_name: Optional[str] = None
    timestamp: str
    text: str


class ExportDocxRequest(BaseModel):
    title: str = "Transkrip Kuliah CapScribe"
    blocks: list[ExportBlock]
    summary: str = ""


@app.post("/api/export/docx")
async def api_export_docx(req: ExportDocxRequest, _: bool = Depends(verify_auth)):
    doc = Document()

    title_p = doc.add_heading(req.title or "Transkrip Kuliah CapScribe", level=1)
    title_p.runs[0].font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    meta = doc.add_paragraph(f"Dibuat oleh CapScribe • {datetime.now().strftime('%d %b %Y, %H:%M')}")
    meta.runs[0].font.size = Pt(9.5)
    meta.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    if req.summary.strip():
        doc.add_heading("Ringkasan Belajar", level=2)
        for line in req.summary.strip().splitlines():
            clean_line = line.replace("### ", "").strip()
            if clean_line:
                p = doc.add_paragraph(clean_line)
                p.paragraph_format.space_after = Pt(4)

    doc.add_heading("Transkrip Lengkap", level=2)
    for b in req.blocks:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        header = f"[{b.timestamp}] " + (f"{b.speaker_name}: " if b.speaker_name else "")
        run_head = p.add_run(header)
        run_head.bold = True
        run_head.font.color.rgb = RGBColor(0x1D, 0x4E, 0xD8)
        run_body = p.add_run(b.text)
        run_body.font.size = Pt(10.5)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="CapScribe_{stamp}.docx"'},
    )
