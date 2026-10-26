"""
backend/config.py

Centralized configuration module for the Hallucination Detection Framework.
Defines model identifiers, file system paths, confidence thresholds, and server settings.
"""

import os
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """
    Application settings and hyperparameters for offline hallucination detection.
    """
    
    # -------------------------------------------------------------------------
    # Base Directories
    # -------------------------------------------------------------------------
    BACKEND_DIR: Path = Path(__file__).resolve().parent
    ROOT_DIR: Path = BACKEND_DIR.parent
    DATA_DIR: Path = ROOT_DIR / "data"
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma_db"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    LOGS_DIR: Path = DATA_DIR / "logs"

    # -------------------------------------------------------------------------
    # Server & API Configuration
    # -------------------------------------------------------------------------
    APP_NAME: str = "Hallucination Detection Framework"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:5500",  # Live Server default
        "*"
    ]

    # -------------------------------------------------------------------------
    # ML & NLP Model Identifiers (100% Offline / Local Hugging Face & spaCy)
    # -------------------------------------------------------------------------
    # Target LLM for local logit & entropy extraction
    TARGET_LLM_MODEL: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Sentence Embeddings for DPR & Knowledge Vault
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Natural Language Inference (NLI) Cross-Encoder
    NLI_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-base"
    
    # spaCy pipeline for sentence tokenization and dependency parsing
    SPACY_MODEL: str = "en_core_web_sm"

    # -------------------------------------------------------------------------
    # Pipeline Hyperparameters & Classification Thresholds
    # -------------------------------------------------------------------------
    # Vector DB Search parameters
    VECTOR_TOP_K: int = 3
    CHROMA_COLLECTION_NAME: str = "ground_truth_knowledge"
    
    # Document chunking defaults
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Token & Logit Confidence Thresholds
    # Below this log-probability or confidence, a token/sentence is marked uncertain
    LOW_CONFIDENCE_THRESHOLD: float = 0.40
    HIGH_CONFIDENCE_THRESHOLD: float = 0.75
    
    # Semantic Entropy Threshold for Hallucination Risk
    HIGH_ENTROPY_THRESHOLD: float = 0.65

    # NLI Cross-Encoder Classification Decision Boundaries
    # Label index mapping typical to DeBERTa NLI: 0 -> Contradiction, 1 -> Entailment, 2 -> Neutral
    NLI_ENTAILMENT_THRESHOLD: float = 0.60
    NLI_CONTRADICTION_THRESHOLD: float = 0.50

    # -------------------------------------------------------------------------
    # Logging Configuration
    # -------------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_FILE_NAME: str = "hallucination_engine.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Instantiate singleton settings object
settings = AppSettings()

# Ensure critical runtime directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)