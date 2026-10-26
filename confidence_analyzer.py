"""
backend/pipeline/confidence_analyzer.py

Module for Step 1 of the Hallucination Detection Pipeline.
Extracts token-level logits, computes transition log-probabilities,
and evaluates sequence perplexity and token-level semantic entropy.
"""

import math
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from backend.config import settings
from backend.utils.logger import logger


class ConfidenceAnalyzer:
    """
    Evaluates LLM generation uncertainty and token-level confidence scores.
    """

    def __init__(self, model_name: str = settings.TARGET_LLM_MODEL):
        """
        Initializes the tokenizer and causal language model offline.

        :param model_name: Hugging Face model identifier or local directory path.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading ConfidenceAnalyzer model '{self.model_name}' on device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=False,
                trust_remote_code=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=False
            )
            if self.device == "cpu":
                self.model.to("cpu")
            self.model.eval()
            logger.info("ConfidenceAnalyzer model loaded and set to evaluation mode.")
        except Exception as err:
            logger.error(f"Failed to load causal LM '{self.model_name}': {err}", exc_info=True)
            raise err

    @torch.inference_mode()
    def analyze_confidence(
        self,
        prompt: Optional[str] = "",
        response_text: str = ""
    ) -> Dict[str, Any]:
        """
        Calculates token probabilities, semantic entropy, and sequence perplexity.

        :param prompt: Optional user input or system prompt.
        :param response_text: The LLM-generated output text to evaluate.
        :return: Structured dictionary containing token metrics and aggregate scores.
        """
        if not response_text.strip():
            logger.warning("Empty response text provided to confidence analyzer.")
            return {
                "mean_confidence": 0.0,
                "mean_entropy": 0.0,
                "perplexity": 0.0,
                "token_count": 0,
                "tokens": [],
                "low_confidence_flag": True
            }

        # Format full sequence and handle prompt boundary
        full_text = f"{prompt.strip()}\n{response_text.strip()}" if prompt.strip() else response_text.strip()
        
        # Tokenize entire context
        encoded_full = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        input_ids = encoded_full.input_ids
        
        # If a prompt exists, identify where the generated response starts
        if prompt.strip():
            encoded_prompt = self.tokenizer(prompt.strip(), return_tensors="pt")
            prompt_token_len = encoded_prompt.input_ids.shape[1]
        else:
            prompt_token_len = 0

        # Forward pass to retrieve logits
        outputs = self.model(input_ids)
        logits = outputs.logits  # Shape: (1, seq_len, vocab_size)

        # Shift logits and targets for next-token prediction evaluation
        # Logits at position i predict token at position i+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Isolate token positions belonging specifically to the response
        response_start_idx = max(0, prompt_token_len - 1)
        response_logits = shift_logits[0, response_start_idx:, :]
        response_targets = shift_labels[0, response_start_idx:]

        if response_targets.shape[0] == 0:
            logger.warning("Response token slice length is zero.")
            return {
                "mean_confidence": 1.0,
                "mean_entropy": 0.0,
                "perplexity": 1.0,
                "token_count": 0,
                "tokens": [],
                "low_confidence_flag": False
            }

        # Compute softmax probabilities and log-softmax across vocabulary
        probs = F.softmax(response_logits, dim=-1)
        log_probs = F.log_softmax(response_logits, dim=-1)

        # 1. Target Token Probabilities & Log-Probabilities
        target_log_probs = log_probs.gather(dim=-1, index=response_targets.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=response_targets.unsqueeze(-1)).squeeze(-1)

        # 2. Shannon / Semantic Entropy: H(X) = -sum(P(x) * log(P(x)))
        # Using clamp to prevent log(0) NaN artifacts
        clamped_probs = torch.clamp(probs, min=1e-12)
        token_entropies = -(clamped_probs * torch.log(clamped_probs)).sum(dim=-1)

        # Convert tensors to CPU lists
        target_probs_list = target_probs.tolist()
        target_log_probs_list = target_log_probs.tolist()
        token_entropies_list = token_entropies.tolist()
        target_ids_list = response_targets.tolist()

        token_details: List[Dict[str, Any]] = []
        low_confidence_tokens = 0

        for idx, (tok_id, prob, log_p, entropy) in enumerate(
            zip(target_ids_list, target_probs_list, target_log_probs_list, token_entropies_list)
        ):
            decoded_token = self.tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
            is_low_confidence = prob < settings.LOW_CONFIDENCE_THRESHOLD
            is_high_entropy = entropy > settings.HIGH_ENTROPY_THRESHOLD

            if is_low_confidence:
                low_confidence_tokens += 1

            token_details.append({
                "token_index": idx,
                "token_id": int(tok_id),
                "token_text": decoded_token,
                "probability": round(float(prob), 4),
                "log_probability": round(float(log_p), 4),
                "entropy": round(float(entropy), 4),
                "is_low_confidence": is_low_confidence,
                "is_high_entropy": is_high_entropy
            })

        # Aggregated Sequence Metrics
        total_tokens = len(token_details)
        mean_prob = float(torch.mean(target_probs).item()) if total_tokens > 0 else 0.0
        mean_entropy = float(torch.mean(token_entropies).item()) if total_tokens > 0 else 0.0
        
        # Perplexity = exp(-mean(log_prob))
        mean_neg_log_prob = -float(torch.mean(target_log_probs).item()) if total_tokens > 0 else 0.0
        perplexity = float(math.exp(min(mean_neg_log_prob, 100.0)))

        return {
            "mean_confidence": round(mean_prob, 4),
            "mean_entropy": round(mean_entropy, 4),
            "perplexity": round(perplexity, 4),
            "token_count": total_tokens,
            "low_confidence_token_count": low_confidence_tokens,
            "low_confidence_ratio": round(low_confidence_tokens / max(1, total_tokens), 4),
            "tokens": token_details,
            "low_confidence_flag": mean_prob < settings.LOW_CONFIDENCE_THRESHOLD
        }