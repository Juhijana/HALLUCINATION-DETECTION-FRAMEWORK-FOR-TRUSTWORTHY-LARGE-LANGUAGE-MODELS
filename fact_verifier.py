"""
backend/pipeline/fact_verifier.py

Module for Step 3 of the Hallucination Detection Pipeline.
Implements Dense Passage Retrieval grounding and Natural Language Inference (NLI)
Cross-Encoder classification to verify atomic claims against local knowledge.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from backend.config import settings
from backend.utils.logger import logger
from backend.storage.vector_vault import VectorVault


class FactVerifier:
    """
    Evaluates factual alignment of atomic claims using Dense Retrieval and Cross-Encoder NLI.
    """

    # Label index mappings for standard DeBERTa/RoBERTa NLI cross-encoders
    # Model config typical order: 0: Contradiction, 1: Entailment, 2: Neutral
    # Fallback mappings handle different MNLI label definitions dynamically.
    NLI_LABEL_MAPPING = {
        0: "CONTRADICTED",
        1: "ENTAILED",
        2: "NEUTRAL"
    }

    def __init__(
        self,
        nli_model_name: str = settings.NLI_MODEL_NAME,
        vector_vault: Optional[VectorVault] = None,
        entailment_thresh: float = settings.NLI_ENTAILMENT_THRESHOLD,
        contradiction_thresh: float = settings.NLI_CONTRADICTION_THRESHOLD
    ):
        """
        Initializes the NLI model and links to the Vector Vault instance.

        :param nli_model_name: Hugging Face model identifier for NLI Cross-Encoder.
        :param vector_vault: Initialized VectorVault client instance.
        :param entailment_thresh: Decision probability boundary for Entailment.
        :param contradiction_thresh: Decision probability boundary for Contradiction.
        """
        self.model_name = nli_model_name
        self.vector_vault = vector_vault
        self.entailment_thresh = entailment_thresh
        self.contradiction_thresh = contradiction_thresh

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading NLI FactVerifier model '{self.model_name}' on device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=False
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=False
            )
            if self.device == "cpu":
                self.model.to("cpu")
            self.model.eval()

            # Align label IDs from model config if explicitly defined
            self._init_label_mapping()
            logger.info("NLI FactVerifier model loaded successfully.")
        except Exception as err:
            logger.error(f"Failed to load NLI model '{self.model_name}': {err}", exc_info=True)
            raise err

    def _init_label_mapping(self) -> None:
        """
        Inspects model config to align integer label indices with canonical NLI labels.
        """
        id2label = getattr(self.model.config, "id2label", None)
        if id2label and isinstance(id2label, dict):
            mapping = {}
            for idx, raw_label in id2label.items():
                lbl = str(raw_label).upper()
                if "ENTAIL" in lbl:
                    mapping[int(idx)] = "ENTAILED"
                elif "CONTRADICT" in lbl:
                    mapping[int(idx)] = "CONTRADICTED"
                else:
                    mapping[int(idx)] = "NEUTRAL"
            if len(mapping) == 3:
                self.NLI_LABEL_MAPPING = mapping
                logger.info(f"Dynamically mapped NLI labels from config: {self.NLI_LABEL_MAPPING}")

    @torch.inference_mode()
    def classify_pair(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Runs cross-encoder NLI inference on a single Premise-Hypothesis pair.

        :param premise: Ground-truth reference context.
        :param hypothesis: Extracted atomic claim to verify.
        :return: Classification verdict, confidence scores, and raw probabilities.
        """
        if not premise.strip() or not hypothesis.strip():
            return {
                "verdict": "NEUTRAL",
                "probabilities": {"entailed": 0.0, "contradicted": 0.0, "neutral": 1.0},
                "confidence": 0.0
            }

        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0)

        prob_dict: Dict[str, float] = {}
        for idx, score in enumerate(probs):
            label_name = self.NLI_LABEL_MAPPING.get(idx, "NEUTRAL").lower()
            prob_dict[label_name] = round(float(score.item()), 4)

        p_entailed = prob_dict.get("entailed", 0.0)
        p_contradicted = prob_dict.get("contradicted", 0.0)
        p_neutral = prob_dict.get("neutral", 0.0)

        # Decision Boundary Logic
        if p_contradicted >= self.contradiction_thresh and p_contradicted > p_entailed:
            verdict = "CONTRADICTED"
            confidence = p_contradicted
        elif p_entailed >= self.entailment_thresh and p_entailed > p_contradicted:
            verdict = "ENTAILED"
            confidence = p_entailed
        else:
            verdict = "NEUTRAL"
            confidence = p_neutral

        return {
            "verdict": verdict,
            "probabilities": {
                "entailed": p_entailed,
                "contradicted": p_contradicted,
                "neutral": p_neutral
            },
            "confidence": round(confidence, 4)
        }

    def verify_claims(
        self,
        claims: List[Dict[str, Any]],
        top_k: int = settings.VECTOR_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Queries the local vector vault for each claim, runs NLI, and aggregates evidence.

        :param claims: List of parsed claim dictionaries from ClaimExtractor.
        :param top_k: Maximum evidence passages to retrieve per claim.
        :return: Augmented claim dictionaries with evidence passages and verification status.
        """
        verified_results: List[Dict[str, Any]] = []

        for item in claims:
            claim_text = item.get("claim_text", "")
            
            # 1. Retrieve DPR context from Vector Vault
            evidence_passages = []
            if self.vector_vault and self.vector_vault.count() > 0:
                evidence_passages = self.vector_vault.query_similarity(query_text=claim_text, top_k=top_k)

            # 2. Evaluate claim against retrieved evidence
            best_verdict = "NEUTRAL"
            best_confidence = 0.0
            best_evidence: Optional[Dict[str, Any]] = None
            all_pair_evaluations: List[Dict[str, Any]] = []

            if evidence_passages:
                for passage in evidence_passages:
                    premise = passage["text"]
                    pair_result = self.classify_pair(premise=premise, hypothesis=claim_text)
                    
                    eval_record = {
                        "evidence_text": premise,
                        "source": passage.get("metadata", {}).get("source", "unknown"),
                        "similarity": passage.get("similarity", 0.0),
                        "verdict": pair_result["verdict"],
                        "confidence": pair_result["confidence"],
                        "probabilities": pair_result["probabilities"]
                    }
                    all_pair_evaluations.append(eval_record)

                    # Prioritize finding explicit contradiction (hallucination discovery)
                    if pair_result["verdict"] == "CONTRADICTED" and pair_result["confidence"] > best_confidence:
                        best_verdict = "CONTRADICTED"
                        best_confidence = pair_result["confidence"]
                        best_evidence = eval_record
                    elif pair_result["verdict"] == "ENTAILED" and best_verdict != "CONTRADICTED":
                        if pair_result["confidence"] > best_confidence:
                            best_verdict = "ENTAILED"
                            best_confidence = pair_result["confidence"]
                            best_evidence = eval_record

                # If no strong entailment/contradiction found, default to highest similarity passage
                if not best_evidence and all_pair_evaluations:
                    best_evidence = all_pair_evaluations[0]
                    best_verdict = best_evidence["verdict"]
                    best_confidence = best_evidence["confidence"]
            else:
                # No evidence retrieved from vault
                best_evidence = None
                best_verdict = "NEUTRAL"
                best_confidence = 0.0

            claim_record = {
                **item,
                "verification_status": best_verdict,
                "verification_confidence": best_confidence,
                "best_evidence": best_evidence,
                "evidence_candidates": all_pair_evaluations
            }
            verified_results.append(claim_record)

        logger.info(f"Completed NLI verification for {len(verified_results)} claims.")
        return verified_results

    def verify_without_vault(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback verification handler when the vector vault is offline or empty.
        """
        results: List[Dict[str, Any]] = []
        for item in claims:
            results.append({
                **item,
                "verification_status": "NEUTRAL",
                "verification_confidence": 0.0,
                "best_evidence": None,
                "evidence_candidates": [],
                "note": "Vector Vault empty or ungrounded."
            })
        return results