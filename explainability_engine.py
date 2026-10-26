"""
backend/pipeline/explainability_engine.py

Module for Step 4 of the Hallucination Detection Pipeline.
Synthesizes token-level uncertainty and NLI fact-verification results into
color-coded character heatmaps, overall trust metrics, and actionable diagnostics.
"""

from typing import List, Dict, Any, Optional
from backend.config import settings
from backend.utils.logger import logger


class ExplainabilityEngine:
    """
    Synthesizes model uncertainty and factual grounding into visual explainability metrics.
    """

    def __init__(
        self,
        low_confidence_threshold: float = settings.LOW_CONFIDENCE_THRESHOLD,
        high_entropy_threshold: float = settings.HIGH_ENTROPY_THRESHOLD
    ):
        """
        :param low_confidence_threshold: Probability threshold below which a token is marked uncertain.
        :param high_entropy_threshold: Entropy cutoff above which generation variance is high.
        """
        self.low_confidence_threshold = low_confidence_threshold
        self.high_entropy_threshold = high_entropy_threshold

    def synthesize_report(
        self,
        full_text: str,
        token_metrics: Dict[str, Any],
        verified_claims: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merges token uncertainty analytics and NLI claim statuses into an explainability payload.

        :param full_text: The complete LLM-generated response.
        :param token_metrics: Output dictionary from ConfidenceAnalyzer (Step 1).
        :param verified_claims: Output list from FactVerifier (Step 3).
        :return: Comprehensive structured report containing scores, heatmap spans, and diagnostics.
        """
        logger.info("Synthesizing visual explainability report...")

        # 1. Calculate claim distribution statistics
        total_claims = len(verified_claims)
        entailed_count = sum(1 for c in verified_claims if c.get("verification_status") == "ENTAILED")
        contradicted_count = sum(1 for c in verified_claims if c.get("verification_status") == "CONTRADICTED")
        neutral_count = sum(1 for c in verified_claims if c.get("verification_status") == "NEUTRAL")

        # 2. Derive composite Trust Score (0 - 100%)
        # Base confidence weighted by factual verification ratios
        mean_token_confidence = token_metrics.get("mean_confidence", 0.75)
        
        if total_claims > 0:
            factuality_ratio = (entailed_count + (0.5 * neutral_count)) / total_claims
            # Explicit contradictions heavily penalize the trust score
            penalty = (contradicted_count / total_claims) * 0.60
            computed_trust = max(0.0, min(1.0, (0.4 * mean_token_confidence) + (0.6 * factuality_ratio) - penalty))
        else:
            computed_trust = mean_token_confidence

        trust_score_pct = round(computed_trust * 100, 2)

        # 3. Formulate high-level Risk Verdict
        if contradicted_count > 0:
            overall_verdict = "HIGH_HALLUCINATION_RISK"
            badge_color = "red"
        elif neutral_count > entailed_count or token_metrics.get("low_confidence_flag", False):
            overall_verdict = "UNCERTAIN_UNVERIFIED"
            badge_color = "yellow"
        else:
            overall_verdict = "TRUSTWORTHY_GROUNDED"
            badge_color = "green"

        # 4. Generate highlighted text spans for front-end rendering
        highlight_spans = self._build_highlight_spans(full_text, verified_claims, token_metrics)

        # 5. Compile synthesis summary
        report: Dict[str, Any] = {
            "overall_trust_score": trust_score_pct,
            "overall_verdict": overall_verdict,
            "badge_color": badge_color,
            "summary_metrics": {
                "total_claims": total_claims,
                "entailed_claims": entailed_count,
                "contradicted_claims": contradicted_count,
                "neutral_claims": neutral_count,
                "mean_token_confidence": round(mean_token_confidence, 4),
                "mean_entropy": token_metrics.get("mean_entropy", 0.0),
                "sequence_perplexity": token_metrics.get("perplexity", 1.0),
                "low_confidence_tokens": token_metrics.get("low_confidence_token_count", 0)
            },
            "highlighted_spans": highlight_spans,
            "claims": verified_claims,
            "token_details": token_metrics.get("tokens", [])
        }

        logger.info(f"Report generated: Status={overall_verdict}, Trust Score={trust_score_pct}%")
        return report

    def _build_highlight_spans(
        self,
        full_text: str,
        claims: List[Dict[str, Any]],
        token_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Maps claims and token positions to character-level highlight spans.
        Status types: 'entailed' (green), 'uncertain' (yellow), 'contradicted' (red).
        """
        spans: List[Dict[str, Any]] = []

        if claims:
            for claim in claims:
                status = claim.get("verification_status", "NEUTRAL")
                conf = claim.get("verification_confidence", 0.0)

                if status == "CONTRADICTED":
                    color_type = "contradicted"  # Red
                elif status == "ENTAILED":
                    color_type = "entailed"      # Green
                else:
                    color_type = "uncertain"     # Yellow

                spans.append({
                    "claim_id": claim.get("claim_id"),
                    "start": claim.get("start_char", 0),
                    "end": claim.get("end_char", len(full_text)),
                    "text": claim.get("claim_text", ""),
                    "color_type": color_type,
                    "confidence": conf,
                    "reasoning": f"NLI Status: {status} (Confidence: {round(conf * 100, 1)}%)"
                })
        else:
            # Fallback to token-level segmentation if no atomic claims were extracted
            for tok in token_metrics.get("tokens", []):
                is_low = tok.get("is_low_confidence", False)
                spans.append({
                    "token_index": tok.get("token_index"),
                    "text": tok.get("token_text"),
                    "color_type": "uncertain" if is_low else "entailed",
                    "confidence": tok.get("probability", 1.0),
                    "reasoning": f"Token Log-Prob Probability: {tok.get('probability')}"
                })

        return spans