# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.1.0
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    device: Literal["auto", "cpu", "cuda:0"] = Field(
        default="auto", description="Device to use for computation"
    )
    qdrant_path: str = Field(
        default="./qdrant_db", description="Path to the local Qdrant database"
    )
    collection_main: str = Field(
        default="rag_multimodal", description="Main collection name in Qdrant"
    )
    collection_mem: str = Field(
        default="agent_memory", description="Memory collection name in Qdrant"
    )
    dense_dim_text: int = Field(
        default=1024, description="Dimension of dense text vectors"
    )
    dense_dim_image: int = Field(
        default=512, description="Dimension of dense image vectors"
    )
