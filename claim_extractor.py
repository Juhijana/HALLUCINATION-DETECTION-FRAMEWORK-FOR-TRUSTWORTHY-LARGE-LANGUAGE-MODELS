"""
backend/pipeline/claim_extractor.py

Module for Step 2 of the Hallucination Detection Pipeline.
Decomposes LLM responses into discrete, testable atomic claims using
spaCy syntactic dependency parsing, clause resolution, and entity recognition.
Includes robust Markdown sanitization and grammatical predicate validation.
"""

import re
from typing import List, Dict, Any, Tuple
import spacy
from spacy.tokens import Doc, Span

from backend.config import settings
from backend.utils.logger import logger


class ClaimExtractor:
    """
    Decomposes natural language paragraphs and formatted Markdown responses
    into atomic factual statements suitable for NLI verification.
    """

    def __init__(self, spacy_model: str = settings.SPACY_MODEL):
        """
        Loads the offline spaCy language model.

        :param spacy_model: The name or local directory path of the spaCy model.
        """
        self.model_name = spacy_model
        logger.info(f"Loading ClaimExtractor with spaCy model: '{self.model_name}'")
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info("spaCy pipeline loaded successfully.")
        except Exception as err:
            logger.error(
                f"Failed to load spaCy model '{self.model_name}'. Ensure it is downloaded: "
                f"python -m spacy download {self.model_name}",
                exc_info=True
            )
            raise err

    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Parses text into a list of atomic claims with character offsets and extracted entities.
        Sanitizes Markdown headers, lists, and non-clausal artifacts.

        :param text: Raw generated string from the LLM.
        :return: List of structured claim dictionaries matching the pipeline interface.
        """
        if not text or not text.strip():
            logger.warning("Empty text string provided for claim extraction.")
            return []

        doc: Doc = self.nlp(text)
        claims: List[Dict[str, Any]] = []
        claim_id_counter = 0

        for sent_idx, sent in enumerate(doc.sents):
            raw_sent_text = sent.text.strip()
            if not raw_sent_text or len(raw_sent_text) < 4:
                continue

            # Attempt to split compound/coordinate clauses into sub-claims
            sub_clauses = self._split_coordinate_clauses(sent)

            for clause_span, is_split in sub_clauses:
                # 1. Clean markdown artifacts and calculate adjusted character offsets
                cleaned_text, start_offset, end_offset = self._clean_markdown_and_offsets(clause_span)

                # 2. Discard headers, pure labels, or strings lacking grammatical clauses
                if not self._is_valid_claim_clause(cleaned_text):
                    continue

                # 3. Extract named entities on the sanitized claim
                claim_doc = self.nlp(cleaned_text)
                entities = [
                    {"text": ent.text, "label": ent.label_}
                    for ent in claim_doc.ents
                ]

                claims.append({
                    "claim_id": f"claim_{claim_id_counter}",
                    "sentence_index": sent_idx,
                    "claim_text": cleaned_text,
                    "start_char": start_offset,
                    "end_char": end_offset,
                    "entities": entities,
                    "is_compound_split": is_split
                })
                claim_id_counter += 1

        logger.info(f"Extracted {len(claims)} valid atomic claims from text input.")
        return claims

    def _split_coordinate_clauses(self, sent: Span) -> List[Tuple[Span, bool]]:
        """
        Splits compound sentences at coordinating conjunctions (CCONJ) only when
        both sides form complete, independent clauses with explicit subjects and predicates.
        Prevents dangling fragments like prepositional or verb lists.
        """
        split_token_indices = []

        for token in sent:
            # Check for coordinating conjunctions linking full independent clauses
            if token.pos_ == "CCONJ" and token.dep_ in ("cc", "conj"):
                conjunct_head = token.head
                
                # Verify right-hand conjunct is a full verbal predicate
                if conjunct_head.pos_ in ("VERB", "AUX"):
                    # Confirm right-hand conjunct has an explicit syntactic subject
                    has_explicit_subj = any(
                        child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass")
                        for child in conjunct_head.children
                    )
                    if has_explicit_subj and token.i > sent.start:
                        split_token_indices.append(token.i)

        if not split_token_indices:
            return [(sent, False)]

        clauses: List[Tuple[Span, bool]] = []
        curr_start = sent.start

        for split_idx in split_token_indices:
            if split_idx > curr_start:
                clause_span = sent.doc[curr_start:split_idx]
                if len(clause_span.text.strip()) > 3:
                    clauses.append((clause_span, True))
                curr_start = split_idx + 1

        if curr_start < sent.end:
            tail_span = sent.doc[curr_start:sent.end]
            if len(tail_span.text.strip()) > 3:
                clauses.append((tail_span, True))

        return clauses if clauses else [(sent, False)]

    def _clean_markdown_and_offsets(self, span: Span) -> Tuple[str, int, int]:
        """
        Strips leading bullet points, Markdown headings, numbering prefixes, 
        bold/italic markers, and whitespace while maintaining character offsets.
        """
        raw_text = span.text
        start_char = span.start_char
        end_char = span.end_char

        # Remove leading whitespace
        l_stripped = raw_text.lstrip()
        start_char += (len(raw_text) - len(l_stripped))

        # Remove trailing whitespace
        r_stripped = l_stripped.rstrip()
        end_char = start_char + len(r_stripped)

        cleaned = r_stripped

        # Strip markdown list markers, numbering, and heading symbols: '* ', '- ', '1. ', '## ', '1) '
        lead_pattern = r"^([*\-#>]+|\d+[\.\)]|\b\d+\b)\s*"
        match_lead = re.match(lead_pattern, cleaned)
        if match_lead:
            shift = match_lead.end()
            cleaned = cleaned[shift:]
            start_char += shift

        # Strip enclosing markdown formatting (**bold**, *italic*, __underline__)
        while (cleaned.startswith("**") and cleaned.endswith("**")) or (cleaned.startswith("__") and cleaned.endswith("__")):
            if len(cleaned) >= 4:
                cleaned = cleaned[2:-2]
                start_char += 2
                end_char -= 2
            else:
                break

        while (cleaned.startswith("*") and cleaned.endswith("*")) or (cleaned.startswith("_") and cleaned.endswith("_")):
            if len(cleaned) >= 2 and not cleaned.startswith("**"):
                cleaned = cleaned[1:-1]
                start_char += 1
                end_char -= 1
            else:
                break

        # Remove inline bold/italic markers and standalone bullet tags
        cleaned_str = re.sub(r"\*\*|__|\*", "", cleaned).strip()

        # If the sentence starts with a label followed by a colon or number prefix, strip it
        # Example: "Fatal-crash involvement: In the US..." -> "In the US..."
        cleaned_str = re.sub(r"^[A-Za-z0-9\s\-]+:\s*", "", cleaned_str)

        return cleaned_str, start_char, end_char

    def _is_valid_claim_clause(self, text: str) -> bool:
        """
        Validates that a text fragment is a factual assertion rather than
        a section header, title, bullet label, or incomplete dangling phrase.
        """
        cleaned = text.strip()
        if not cleaned:
            return False

        # Discard strings ending with colons (e.g., "Fatal Crashes & Severity:")
        if cleaned.endswith(":"):
            return False

        doc = self.nlp(cleaned)

        # Discard fragments with fewer than 4 tokens
        if len(doc) < 4:
            return False

        has_predicate = False
        has_subject = False

        for token in doc:
            # Check for verbal predicate (VERB or AUX)
            if token.pos_ in ("VERB", "AUX"):
                has_predicate = True

            # Check for grammatical subject
            if token.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl"):
                has_subject = True

        # Candidate claim must contain both a subject and a verb predicate
        return has_predicate and has_subject