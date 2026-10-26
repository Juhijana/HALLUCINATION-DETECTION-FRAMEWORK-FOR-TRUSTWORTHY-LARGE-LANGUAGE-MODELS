"""
backend/pipeline/nlp_preprocessor.py

Linguistic Preprocessing & Analysis Module covering syllabus topics:
- Tokenization (Word and Sentence level)
- Stop words removal (spaCy / NLTK)
- Comparative Stemming (NLTK PorterStemmer) vs Lemmatization (spaCy)
- Part-of-Speech (POS) Tagging & Named Entity Recognition (NER)
- Word Frequency Distribution
"""

from typing import Dict, Any, List
from collections import Counter
import spacy
from nltk.stem import PorterStemmer

from backend.config import settings
from backend.utils.logger import logger


class NLPPreprocessor:
    """
    Executes standard linguistic preprocessing and feature analysis on input text.
    """

    def __init__(self, spacy_model: str = settings.SPACY_MODEL):
        logger.info(f"Initializing NLPPreprocessor with {spacy_model} and NLTK PorterStemmer...")
        try:
            self.nlp = spacy.load(spacy_model)
            self.stemmer = PorterStemmer()
            logger.info("NLPPreprocessor initialized successfully.")
        except Exception as err:
            logger.error(f"Failed to initialize NLPPreprocessor: {err}", exc_info=True)
            raise err

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Executes tokenization, stop word filtering, POS tagging, 
        entity extraction, and stemming vs. lemmatization comparisons.
        """
        if not text or not text.strip():
            return {}

        doc = self.nlp(text)

        # 1. Tokenization & Word Frequency Distribution
        raw_tokens = [token.text for token in doc if not token.is_space]
        word_freq = dict(Counter([t.lower() for t in raw_tokens if t.isalnum()]).most_common(10))

        # 2. Stop Words Identification & Filtering
        stop_words_found = [token.text for token in doc if token.is_stop]
        filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

        # 3. Stemming vs. Lemmatization Comparison
        stem_vs_lemma: List[Dict[str, str]] = []
        seen_words = set()

        for token in doc:
            word = token.text.strip()
            # Pick non-stop, alphabetic words to show meaningful linguistic root comparison
            if word.isalpha() and not token.is_stop and word.lower() not in seen_words:
                seen_words.add(word.lower())
                stem_vs_lemma.append({
                    "original": word,
                    "stemmed_nltk": self.stemmer.stem(word),
                    "lemmatized_spacy": token.lemma_,
                    "pos_tag": token.pos_
                })
            if len(stem_vs_lemma) >= 15:
                break

        # 4. Part-of-Speech (POS) Tag Summary
        pos_summary = dict(Counter([token.pos_ for token in doc if not token.is_space]))

        # 5. Named Entity Recognition (NER)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        return {
            "tokenization": {
                "total_tokens": len(raw_tokens),
                "unique_words": len(set(t.lower() for t in raw_tokens if t.isalnum())),
                "top_word_frequencies": word_freq
            },
            "stop_words": {
                "total_stop_words_count": len(stop_words_found),
                "filtered_content_words_count": len(filtered_tokens),
                "sample_removed_stop_words": list(set(stop_words_found))[:10]
            },
            "stemming_vs_lemmatization": stem_vs_lemma,
            "part_of_speech_distribution": pos_summary,
            "named_entities": entities
        }