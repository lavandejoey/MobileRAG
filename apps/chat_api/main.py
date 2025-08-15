# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.12.0
"""

from fastapi import FastAPI

from apps.chat_api.routes import chat, evidence, ingest, status

app = FastAPI()

app.include_router(chat.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")
app.include_router(evidence.router, prefix="/api")
app.include_router(status.router, prefix="/api")
