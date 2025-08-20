# -*- coding: utf-8 -*-
"""
@file: apps/chat_api/routes/__init__.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.12.0
"""

from apps.chat_api.routes.chat import router as chat_router
from apps.chat_api.routes.evidence import router as evidence_router
from apps.chat_api.routes.ingest import router as ingest_router
from apps.chat_api.routes.status import router as status_router

__all__ = ["chat_router", "ingest_router", "evidence_router", "status_router"]
