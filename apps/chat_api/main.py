# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.12.0
"""

from fastapi import FastAPI

from apps.chat_api.routes import chat

app = FastAPI()

app.include_router(chat.router, prefix="/api")
