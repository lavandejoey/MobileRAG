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

import csv
import hashlib
import html
from html.parser import HTMLParser
import mimetypes
from pathlib import Path
from typing import List, Tuple
from xml.etree import ElementTree
from zipfile import ZipFile

from pypdf import PdfReader


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        return "\n".join(self._parts)


class ParsedSection:
    def __init__(self, text: str, source_label: str | None = None) -> None:
        self.text = text
        self.source_label = source_label


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


def read_pdf_sections(path: Path) -> List[ParsedSection]:
    reader = PdfReader(str(path))
    sections: List[ParsedSection] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            sections.append(ParsedSection(text=t.strip(), source_label=f"page {idx}"))
    return sections


def read_docx_sections(path: Path) -> List[ParsedSection]:
    with ZipFile(path) as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paras: List[str] = []
    for para in root.findall(".//w:p", ns):
        texts = [node.text or "" for node in para.findall(".//w:t", ns)]
        joined = "".join(texts).strip()
        if joined:
            paras.append(joined)
    if not paras:
        return []
    return [ParsedSection(text="\n".join(paras))]


def read_html_sections(path: Path) -> List[ParsedSection]:
    parser = _HTMLTextExtractor()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    text = html.unescape(parser.get_text()).strip()
    return [ParsedSection(text=text)] if text else []


def read_csv_sections(path: Path, rows_per_section: int = 40) -> List[ParsedSection]:
    sections: List[ParsedSection] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return sections
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    if not body:
        line = " | ".join(cell.strip() for cell in header if cell.strip())
        return [ParsedSection(text=line)] if line else []
    for start in range(0, len(body), rows_per_section):
        batch = body[start:start + rows_per_section]
        rendered = []
        for offset, row in enumerate(batch, start=start + 2):
            pairs = []
            for idx, cell in enumerate(row):
                key = header[idx].strip() if idx < len(header) and header[idx].strip() else f"col{idx + 1}"
                val = cell.strip()
                if val:
                    pairs.append(f"{key}: {val}")
            if pairs:
                rendered.append(f"row {offset}: " + "; ".join(pairs))
        text = "\n".join(rendered).strip()
        if text:
            row_end = start + 1 + len(batch)
            sections.append(ParsedSection(text=text, source_label=f"rows {start + 2}-{row_end}"))
    return sections


def parse_file_sections(path: Path) -> Tuple[List[ParsedSection], str]:
    mime = guess_mime(path)
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        text = read_text_file(path).strip()
        sections = [ParsedSection(text=text)] if text else []
    elif ext == ".pdf":
        sections = read_pdf_sections(path)
    elif ext == ".docx":
        sections = read_docx_sections(path)
    elif ext in (".html", ".htm"):
        sections = read_html_sections(path)
    elif ext == ".csv":
        sections = read_csv_sections(path)
    else:
        raise ValueError(f"unsupported extension: {ext}")

    sections = [section for section in sections if section.text.strip()]
    if not sections:
        raise ValueError("empty document")
    return sections, mime
