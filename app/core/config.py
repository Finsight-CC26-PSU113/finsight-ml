"""
OCR FinSight - Core Configuration
Pydantic BaseSettings loaded from .env / environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings — override via .env or environment variables."""

    # ── Project paths ──────────────────────────────────────────────────
    ROOT_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = ROOT_DIR / "models"

    # ── API ────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
    CORS_ORIGINS: list[str] = ["*"]

    # ── OCR (PaddleOCR) ────────────────────────────────────────────────
    OCR_GPU: bool = False
    OCR_LANG: str = "latin"

    # ── Classifier V5 (line classification) ────────────────────────────
    CLASSIFIER_V5_WEIGHTS: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "models"
        / "classifier_v5_indonesia"
        / "best_weights.weights.h5"
    )
    CONTEXT_WINDOW: int = 4  # V5 uses ±4 lines

    # ── Category Classifier (transaction category) ─────────────────────
    CATEGORY_MODEL: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "models"
        / "category_classifier"
        / "best_model.keras"
    )
    CATEGORY_CONFIG: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "models"
        / "category_classifier"
        / "config.json"
    )
    CATEGORY_LABEL_MAPPING: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "models"
        / "category_classifier"
        / "label_mapping.json"
    )
    CATEGORY_TOKENIZER: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "models"
        / "category_classifier"
        / "tokenizer.json"
    )

    # ── Image preprocessing ────────────────────────────────────────────
    MAX_IMAGE_SIZE: int = 1920
    UPSCALE_FACTOR: float = 2.0

    # ── Line classes (12 categories) ───────────────────────────────────
    LINE_CLASSES: dict[int, str] = {
        0: "STORE",
        1: "ADDRESS_CONTACT",
        2: "DATE",
        3: "ITEM_DESC",
        4: "ITEM_PRICE/QTY",
        5: "SUBTOTAL",
        6: "TAX",
        7: "DISCOUNT",
        8: "SERVICE_CHARGE",
        9: "GRAND_TOTAL",
        10: "CASH_PAYMENT",
        11: "OTHER",
    }

    # ── Model hyperparameters ──────────────────────────────────────────
    EMBEDDING_DIM: int = 64
    LSTM_UNITS: int = 128
    DENSE_UNITS: int = 64
    DROPOUT_RATE: float = 0.3
    MAX_TEXT_LENGTH: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Singleton
settings = Settings()
