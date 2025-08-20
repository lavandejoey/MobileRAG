# -*- coding: utf-8 -*-
"""
@file: core/config/settings.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.5.0
"""

from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class NamedVectorConfig(BaseSettings):
    size: Optional[int] = None
    distance: Optional[str] = None
    sparse: Optional[bool] = False
    name: Optional[str] = None  # Add name attribute for easier access


class VectorStoreConfig(BaseSettings):
    kind: str = Field(default="qdrant_local", description="Type of vector store")
    local_path: str = Field(
        default=str(ROOT_DIR / "qdrant_db"), description="Path to the local vector store database"
    )
    collection: str = Field(
        default="rag_multimodal", description="Main collection name in vector store"
    )
    named_vectors: Dict[str, NamedVectorConfig] = Field(
        default_factory=lambda: {
            "text_dense": NamedVectorConfig(size=1024, distance="cosine", name="text_dense"),
            "image": NamedVectorConfig(size=512, distance="cosine", name="image"),
            "text_sparse": NamedVectorConfig(sparse=True, name="text_sparse"),
        },
        description="Configuration for named vectors",
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    device: Literal["auto", "cpu", "cuda:0"] = Field(
        default="auto", description="Device to use for computation"
    )
    vectorstore: VectorStoreConfig = Field(
        default_factory=VectorStoreConfig, description="Vector store configuration"
    )
    collection_mem: str = Field(
        default="agent_memory", description="Memory collection name in Qdrant"
    )
    quantization: Literal["none", "4bit", "8bit"] = Field(
        default="none", description="Quantization level for models"
    )
    dense_dim_text: int = Field(default=1024, description="Dimension of dense text vectors")
    dense_dim_image: int = Field(default=512, description="Dimension of dense image vectors")
