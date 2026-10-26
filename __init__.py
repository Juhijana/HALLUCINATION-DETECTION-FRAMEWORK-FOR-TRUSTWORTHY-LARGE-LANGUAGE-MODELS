"""
backend/pipeline/__init__.py

Initialization module for the 4-step Hallucination Detection Pipeline:
1. ConfidenceAnalyzer   - Token-level log-probabilities & semantic entropy extraction.
2. ClaimExtractor        - Dependency parsing into atomic testable claims (spaCy).
3. FactVerifier          - Dense Passage Retrieval (DPR) & NLI Cross-Encoder verification.
4. ExplainabilityEngine  - Score aggregation, heatmap synthesis, and XAI formatting.
"""

from backend.pipeline.confidence_analyzer import ConfidenceAnalyzer
from backend.pipeline.claim_extractor import ClaimExtractor
from backend.pipeline.fact_verifier import FactVerifier
from backend.pipeline.explainability_engine import ExplainabilityEngine

__all__ = [
    "ConfidenceAnalyzer",
    "ClaimExtractor",
    "FactVerifier",
    "ExplainabilityEngine",
]