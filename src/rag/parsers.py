#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filesystem scanning utilities.
src/rag/fs_scan.py

@author: LIU Ziyi
@date: 2026-01-01
@license: Apache-2.0
"""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Tuple

from pypdf import PdfReader


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def file_sha1(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            texts.append(t)
    return "\n\n".join(texts)


def parse_file(path: Path) -> Tuple[str, str]:
    mime = guess_mime(path)
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        text = read_text_file(path)
    elif ext == ".pdf":
        text = read_pdf_file(path)
    else:
        raise ValueError(f"unsupported extension: {ext}")

    text = (text or "").strip()
    if not text:
        raise ValueError("empty document")
    return text, mime
