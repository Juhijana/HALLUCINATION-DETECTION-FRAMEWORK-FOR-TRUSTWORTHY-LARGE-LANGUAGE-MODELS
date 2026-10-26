"""
backend/app.py

Main FastAPI application entry point.
Exposes REST endpoints for the 4-step hallucination detection pipeline,
knowledge base document management, and analytical diagnostic aggregations.
"""

import sys
from pathlib import Path

# Automatically add the project root directory to Python's module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import shutil
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import settings
from backend.utils.logger import logger
from backend.storage.document_loader import DocumentLoader
from backend.storage.vector_vault import VectorVault
from backend.pipeline.confidence_analyzer import ConfidenceAnalyzer
from backend.pipeline.claim_extractor import ClaimExtractor
from backend.pipeline.fact_verifier import FactVerifier
from backend.pipeline.explainability_engine import ExplainabilityEngine
from backend.pipeline.dl_evaluator import DeepLearningEvaluator
from backend.pipeline.nlp_preprocessor import NLPPreprocessor

# -----------------------------------------------------------------------------
# Pydantic Schemas & Data Contracts
# -----------------------------------------------------------------------------
class AnalysisRequest(BaseModel):
    """Payload for full response hallucination analysis."""
    text: str = Field(..., min_length=3, description="LLM-generated text to verify.")
    prompt: Optional[str] = Field(default="", description="Original user prompt context.")
    use_ground_truth: bool = Field(default=True, description="Whether to query Vector Vault.")
    top_k_evidence: int = Field(default=settings.VECTOR_TOP_K, ge=1, le=10)


class DirectFactCheckRequest(BaseModel):
    """Payload for checking an individual claim against a direct premise."""
    premise: str = Field(..., min_length=1)
    hypothesis: str = Field(..., min_length=1)


class DocumentMetadataResponse(BaseModel):
    """Metadata format for indexed knowledge documents."""
    doc_id: str
    filename: str
    chunk_count: int
    created_at: float


# -----------------------------------------------------------------------------
# Application State & Lifespan Handler
# -----------------------------------------------------------------------------
# Global service instances
vault: Optional[VectorVault] = None
loader: Optional[DocumentLoader] = None
confidence_analyzer: Optional[ConfidenceAnalyzer] = None
claim_extractor: Optional[ClaimExtractor] = None
fact_verifier: Optional[FactVerifier] = None
explainability_engine: Optional[ExplainabilityEngine] = None
dl_evaluator: Optional[DeepLearningEvaluator] = None
nlp_preprocessor: Optional[NLPPreprocessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown event handler.
    Initializes vector storage, parses pipelines, and caches models offline.
    """
    global vault, loader, confidence_analyzer, claim_extractor, fact_verifier, explainability_engine, dl_evaluator, nlp_preprocessor
    logger.info("Initializing offline Hallucination Detection services...")

    try:
        loader = DocumentLoader(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        vault = VectorVault(persist_dir=settings.CHROMA_PERSIST_DIR, collection_name=settings.CHROMA_COLLECTION_NAME)
        
        confidence_analyzer = ConfidenceAnalyzer(model_name=settings.TARGET_LLM_MODEL)
        claim_extractor = ClaimExtractor(spacy_model=settings.SPACY_MODEL)
        fact_verifier = FactVerifier(
            nli_model_name=settings.NLI_MODEL_NAME,
            vector_vault=vault,
            entailment_thresh=settings.NLI_ENTAILMENT_THRESHOLD,
            contradiction_thresh=settings.NLI_CONTRADICTION_THRESHOLD
        )
        explainability_engine = ExplainabilityEngine()
        
        # Initialize Deep Learning Evaluator
        dl_evaluator = DeepLearningEvaluator(device="cpu")

        # Initialize NLP Syllabus Preprocessor
        nlp_preprocessor = NLPPreprocessor(spacy_model=settings.SPACY_MODEL)
        
        logger.info("All offline pipeline models and VectorVault successfully loaded.")
    except Exception as e:
        logger.error(f"Critical failure initializing pipeline modules: {str(e)}", exc_info=True)
        raise e

    yield

    logger.info("Shutting down Hallucination Detection framework...")


# -----------------------------------------------------------------------------
# FastAPI App Definition
# -----------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Pipeline REST Endpoints
# -----------------------------------------------------------------------------
@app.post(f"{settings.API_V1_PREFIX}/analyze", status_code=status.HTTP_200_OK)
async def analyze_response(request: AnalysisRequest) -> Dict[str, Any]:
    """
    Full 4-Step Hallucination Detection Pipeline:
    1. Confidence & Logit Extraction
    2. Dependency Parsing Claim Extraction
    3. Grounding & Cross-Encoder Fact Verification
    4. XAI Synthesis & Heatmap Payload Generation
    """
    start_time = time.perf_counter()
    logger.info(f"Received hallucination analysis request ({len(request.text)} characters)")

    try:
        # Step 1: Token confidence and semantic entropy estimation
        confidence_result = confidence_analyzer.analyze_confidence(
            prompt=request.prompt,
            response_text=request.text
        )

        # Step 2: Extract discrete atomic claims via spaCy dependency parsing
        extracted_claims = claim_extractor.extract_claims(text=request.text)

        # Step 3: Verify claims against Vector Vault ground-truth via NLI
        verification_results = []
        if request.use_ground_truth and vault.count() > 0:
            verification_results = fact_verifier.verify_claims(
                claims=extracted_claims,
                top_k=request.top_k_evidence
            )
        else:
            # Fallback if vault is empty or ground truth disabled
            verification_results = fact_verifier.verify_without_vault(claims=extracted_claims)

        # Step 4: Synthesize visual explainability (heatmaps, overall trust score, token alerts)
        xai_payload = explainability_engine.synthesize_report(
            full_text=request.text,
            token_metrics=confidence_result,
            verified_claims=verification_results
        )

        # Deep Learning Multi-Model Evaluation
        if dl_evaluator and extracted_claims:
            eval_list = []
            for claim in extracted_claims:
                c_text = claim.get("claim_text", "")
                eval_list.append({
                    "claim": c_text,
                    "model_evaluations": dl_evaluator.evaluate_claim_representation(c_text)
                })
            xai_payload["neural_baseline_evaluations"] = eval_list

        # Attach NLP preprocessing diagnostics (Tokens, Stopwords, Stemming vs Lemma)
        if nlp_preprocessor:
            xai_payload["nlp_preprocessing_telemetry"] = nlp_preprocessor.analyze_text(request.text)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        xai_payload["execution_time_ms"] = elapsed_ms
        logger.info(f"Analysis completed in {elapsed_ms}ms with Trust Score: {xai_payload.get('overall_trust_score', 0)}")

        return xai_payload

    except Exception as e:
        logger.error(f"Error during analysis pipeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing error: {str(e)}"
        )


@app.post(f"{settings.API_V1_PREFIX}/verify-pair", status_code=status.HTTP_200_OK)
async def verify_pair(request: DirectFactCheckRequest) -> Dict[str, Any]:
    """
    Direct premise-hypothesis classification using the local NLI Cross-Encoder.
    """
    try:
        result = fact_verifier.classify_pair(premise=request.premise, hypothesis=request.hypothesis)
        return result
    except Exception as e:
        logger.error(f"Error checking NLI pair: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# -----------------------------------------------------------------------------
# Knowledge Base & Vector Vault Endpoints
# -----------------------------------------------------------------------------
@app.post(f"{settings.API_V1_PREFIX}/knowledge/upload", status_code=status.HTTP_201_CREATED)
async def upload_knowledge_document(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Saves and indexes a reference text or PDF document into the local ChromaDB vault.
    """
    file_path = settings.UPLOADS_DIR / file.filename
    try:
        # Save file to upload staging directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse and chunk document
        chunks, metadata = loader.load_and_chunk(file_path=file_path, custom_id=doc_id)
        
        # Ingest chunks into ChromaDB
        inserted_count = vault.add_documents(chunks=chunks, metadatas=metadata)
        logger.info(f"Successfully indexed '{file.filename}' ({inserted_count} chunks)")

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": inserted_count,
            "message": f"Successfully indexed {inserted_count} text passages into Vector Vault."
        }
    except Exception as e:
        logger.error(f"Failed to ingest knowledge document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        file.file.close()


@app.get(f"{settings.API_V1_PREFIX}/knowledge/documents", status_code=status.HTTP_200_OK)
async def list_knowledge_documents() -> Dict[str, Any]:
    """
    Lists summary metrics and all documents currently stored in the Vector Vault.
    """
    try:
        docs = vault.get_indexed_documents()
        total_chunks = vault.count()
        return {
            "total_documents": len(docs),
            "total_chunks": total_chunks,
            "documents": docs
        }
    except Exception as e:
        logger.error(f"Failed to retrieve documents: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.delete(f"{settings.API_V1_PREFIX}/knowledge/documents/{{doc_name}}", status_code=status.HTTP_200_OK)
async def delete_knowledge_document(doc_name: str) -> Dict[str, Any]:
    """
    Deletes all chunks corresponding to a specific document from the Vector Vault.
    """
    try:
        deleted_count = vault.delete_by_source(source_name=doc_name)
        return {
            "status": "success",
            "deleted_document": doc_name,
            "chunks_removed": deleted_count
        }
    except Exception as e:
        logger.error(f"Failed to delete document '{doc_name}': {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# -----------------------------------------------------------------------------
# Diagnostics & System Analytics Endpoints
# -----------------------------------------------------------------------------
@app.get(f"{settings.API_V1_PREFIX}/analytics/overview", status_code=status.HTTP_200_OK)
async def get_system_overview() -> Dict[str, Any]:
    """
    Returns framework status, model identifiers, and storage statistics.
    """
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "active_models": {
            "target_llm": settings.TARGET_LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "nli_cross_encoder": settings.NLI_MODEL_NAME,
            "spacy_pipeline": settings.SPACY_MODEL
        },
        "thresholds": {
            "low_confidence": settings.LOW_CONFIDENCE_THRESHOLD,
            "high_entropy": settings.HIGH_ENTROPY_THRESHOLD,
            "nli_entailment": settings.NLI_ENTAILMENT_THRESHOLD,
            "nli_contradiction": settings.NLI_CONTRADICTION_THRESHOLD
        },
        "vault_statistics": {
            "total_chunks": vault.count() if vault else 0
        }
    }


@app.post(f"{settings.API_V1_PREFIX}/analytics/nlp-preprocessing", status_code=status.HTTP_200_OK)
async def inspect_nlp_preprocessing(request: AnalysisRequest) -> Dict[str, Any]:
    """
    Dedicated endpoint exposing Unit-1 NLP preprocessing:
    Tokenization, Stop Word Removal, Stemming (NLTK) vs Lemmatization (spaCy), POS, and NER.
    """
    if not nlp_preprocessor:
        raise HTTPException(status_code=500, detail="NLP Preprocessor not initialized")
    return nlp_preprocessor.analyze_text(request.text)


# -----------------------------------------------------------------------------
# Static Web Files Mount (Frontend Dashboard)
# -----------------------------------------------------------------------------
frontend_dir = settings.ROOT_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")