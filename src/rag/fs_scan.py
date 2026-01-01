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

import glob
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Union, Optional

GlobLike = Union[str, Path]
GlobPatterns = Union[GlobLike, Sequence[GlobLike]]


def _is_abs_glob(p: str) -> bool:
    return (
            os.path.isabs(p)
            or (len(p) >= 2 and p[1] == ":")
            or p.startswith("\\\\")
    )


def _iter_glob(pat: str) -> Iterable[Path]:
    if _is_abs_glob(pat):
        for m in glob.glob(pat, recursive=True):
            yield Path(m)
    else:
        for m in Path().glob(pat):
            yield m


def list_doc_paths(
        patterns: Sequence[str],
        *,
        exts: Optional[Sequence[str]] = None,
        follow_symlinks: bool = False,
        max_file_size_mb: Optional[float] = None,
) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()

    exts_norm = None
    if exts:
        exts_norm = {
            e.lower() if e.startswith(".") else f".{e.lower()}"
            for e in exts
        }

    for pat in patterns:
        if not pat:
            continue

        pat = os.path.expanduser(os.path.expandvars(pat))

        try:
            for p in _iter_glob(pat):
                try:
                    if not p.exists() or not p.is_file():
                        continue
                    if p.is_symlink() and not follow_symlinks:
                        continue
                    if exts_norm and p.suffix.lower() not in exts_norm:
                        continue
                    if max_file_size_mb is not None:
                        if p.stat().st_size > max_file_size_mb * 1024 * 1024:
                            continue

                    rp = p.resolve()
                    k = str(rp)
                    if k in seen:
                        continue

                    seen.add(k)
                    out.append(rp)

                except (OSError, PermissionError):
                    continue
        except (OSError, NotImplementedError):
            continue

    return out
