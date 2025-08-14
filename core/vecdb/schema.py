# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.5.0
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Vector(BaseModel):
    vector: List[float]
    name: Optional[str] = None


class SparseVector(BaseModel):
    indices: List[int]
    values: List[float]


class Point(BaseModel):
    id: str
    vectors: Dict[str, Any]  # Can contain dense and sparse vectors
    payload: Dict[str, Any]
