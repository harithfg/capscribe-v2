"""
CapScribe V2 - Web-Based Accessible Lecture Recording
Backend: FastAPI + Deepgram Nova-3 (diarization + keyterms + code-switching)
         + Groq Whisper (fallback STT) + Groq LLM (study summary)
         + Auto-detect Dosen (pembicara dominan) + PWA
"""

import io
import os
import re
import secrets
import struct
import zlib
from datetime import datetime
from functools import lru_cache
from typing import Optional

import httpx
from docx import Document
from docx.shared import Pt, RGBColor
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
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
MAX_UPLOAD_BYTES = 100 * 10
