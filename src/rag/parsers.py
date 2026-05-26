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
import shutil
import subprocess
import tempfile
from typing import List, Tuple
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

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


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return True
    if b"\x00" in data:
        return False
    allowed = 0
    for b in data:
        if b in (9, 10, 13) or 32 <= b <= 126 or b >= 128:
            allowed += 1
    return (allowed / max(1, len(data))) >= 0.85


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    sample = data[:8192]
    if not _looks_like_text(sample):
        raise ValueError("not a readable text file")
    return data.decode("utf-8", errors="ignore")


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
    try:
        with ZipFile(path) as zf:
            xml_bytes = zf.read("word/document.xml")
    except (BadZipFile, KeyError) as exc:
        raise ValueError("not a readable docx file") from exc
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


def read_xlsx_sections(path: Path, rows_per_section: int = 40) -> List[ParsedSection]:
    try:
        with ZipFile(path) as zf:
            shared_strings: List[str] = []
            if "xl/sharedStrings.xml" in zf.namelist():
                sroot = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
                sns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for si in sroot.findall(".//a:si", sns):
                    parts = [node.text or "" for node in si.findall(".//a:t", sns)]
                    shared_strings.append("".join(parts))

            wb = ElementTree.fromstring(zf.read("xl/workbook.xml"))
            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
            rels_root = ElementTree.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            rels = {
                rel.attrib.get("Id", ""): rel.attrib.get("Target", "")
                for rel in rels_root.findall(".//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
            }

            sections: List[ParsedSection] = []
            for sheet in wb.findall(".//a:sheets/a:sheet", ns):
                sheet_name = sheet.attrib.get("name", "sheet")
                rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
                target = rels.get(rel_id, "")
                if not target:
                    continue
                sheet_path = "xl/" + target.lstrip("/")
                sroot = ElementTree.fromstring(zf.read(sheet_path))
                rows: List[List[str]] = []
                for row in sroot.findall(".//a:sheetData/a:row", ns):
                    rendered: List[str] = []
                    for cell in row.findall("a:c", ns):
                        cell_type = cell.attrib.get("t", "")
                        value_node = cell.find("a:v", ns)
                        raw = value_node.text if value_node is not None and value_node.text is not None else ""
                        if cell_type == "s" and raw.isdigit():
                            idx = int(raw)
                            raw = shared_strings[idx] if 0 <= idx < len(shared_strings) else raw
                        elif cell_type == "inlineStr":
                            raw = "".join(node.text or "" for node in cell.findall(".//a:t", ns))
                        raw = raw.strip()
                        rendered.append(raw)
                    if any(cell for cell in rendered):
                        rows.append(rendered)

                if not rows:
                    continue
                header = rows[0]
                body = rows[1:] if len(rows) > 1 else []
                if not body:
                    line = " | ".join(cell for cell in header if cell)
                    if line:
                        sections.append(ParsedSection(text=line, source_label=sheet_name))
                    continue
                for start in range(0, len(body), rows_per_section):
                    batch = body[start:start + rows_per_section]
                    rendered_rows = []
                    for offset, row in enumerate(batch, start=start + 2):
                        pairs = []
                        for idx, cell in enumerate(row):
                            key = header[idx].strip() if idx < len(header) and str(header[idx]).strip() else f"col{idx + 1}"
                            val = cell.strip()
                            if val:
                                pairs.append(f"{key}: {val}")
                        if pairs:
                            rendered_rows.append(f"row {offset}: " + "; ".join(pairs))
                    text = "\n".join(rendered_rows).strip()
                    if text:
                        end_row = start + 1 + len(batch)
                        sections.append(ParsedSection(text=text, source_label=f"{sheet_name} rows {start + 2}-{end_row}"))
            return sections
    except (BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ValueError("not a readable xlsx file") from exc


def _run_text_converter(cmd: list[str], cwd: Path | None = None) -> str:
    out = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise ValueError((out.stderr or out.stdout or "conversion failed").strip())
    return out.stdout.strip()


def read_legacy_doc_sections(path: Path) -> List[ParsedSection]:
    antiword = shutil.which("antiword")
    if antiword:
        text = _run_text_converter([antiword, str(path)])
        return [ParsedSection(text=text)] if text else []
    catdoc = shutil.which("catdoc")
    if catdoc:
        text = _run_text_converter([catdoc, str(path)])
        return [ParsedSection(text=text)] if text else []
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        with tempfile.TemporaryDirectory(prefix="mobilerag-doc-") as tmp:
            outdir = Path(tmp)
            out = subprocess.run(
                [soffice, "--headless", "--convert-to", "txt:Text", "--outdir", str(outdir), str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if out.returncode == 0:
                txt_path = outdir / f"{path.stem}.txt"
                if txt_path.exists():
                    text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
                    return [ParsedSection(text=text)] if text else []
    raise ValueError("not a readable doc file")


def read_legacy_xls_sections(path: Path) -> List[ParsedSection]:
    xls2csv = shutil.which("xls2csv")
    if xls2csv:
        csv_text = _run_text_converter([xls2csv, str(path)])
        rows = list(csv.reader(csv_text.splitlines()))
        if not rows:
            return []
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".csv", delete=True) as tmp:
            writer = csv.writer(tmp)
            writer.writerows(rows)
            tmp.flush()
            return read_csv_sections(Path(tmp.name))
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        with tempfile.TemporaryDirectory(prefix="mobilerag-xls-") as tmp:
            outdir = Path(tmp)
            out = subprocess.run(
                [soffice, "--headless", "--convert-to", "csv", "--outdir", str(outdir), str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if out.returncode == 0:
                csv_path = outdir / f"{path.stem}.csv"
                if csv_path.exists():
                    return read_csv_sections(csv_path)
    raise ValueError("not a readable xls file")


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
    sections: List[ParsedSection] | None = None
    errors: List[str] = []

    # Prefer explicit handlers first, then fall back to text if readable.
    if ext == ".pdf":
        try:
            sections = read_pdf_sections(path)
        except Exception as exc:
            errors.append(str(exc))
    elif ext == ".docx":
        try:
            sections = read_docx_sections(path)
        except Exception as exc:
            errors.append(str(exc))
    elif ext in (".xlsx", ".xlsm"):
        try:
            sections = read_xlsx_sections(path)
        except Exception as exc:
            errors.append(str(exc))
    elif ext == ".doc":
        try:
            sections = read_legacy_doc_sections(path)
        except Exception as exc:
            errors.append(str(exc))
    elif ext == ".xls":
        try:
            sections = read_legacy_xls_sections(path)
        except Exception as exc:
            errors.append(str(exc))
    elif ext in (".html", ".htm"):
        try:
            sections = read_html_sections(path)
        except Exception as exc:
            errors.append(str(exc))
    elif ext == ".csv":
        try:
            sections = read_csv_sections(path)
        except Exception as exc:
            errors.append(str(exc))

    if sections is None:
        try:
            text = read_text_file(path).strip()
            sections = [ParsedSection(text=text)] if text else []
        except Exception as exc:
            errors.append(str(exc))
            raise ValueError(f"unsupported or unreadable file: {ext or path.name}") from exc

    sections = [section for section in sections if section.text.strip()]
    if not sections:
        raise ValueError("empty document")
    return sections, mime
