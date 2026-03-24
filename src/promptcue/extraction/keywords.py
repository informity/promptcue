# promptcue | KeyBERT-based keyword and keyphrase extractor
# Maintainer: Informity

from __future__ import annotations

import threading
from typing import Any

from promptcue.models.schema import PromptCueKeyword


class PromptCueKeywordExtractor:
    """Extracts keywords and keyphrases from a query using KeyBERT.

    KeyBERT is loaded lazily on the first extract() call.  A threading.Lock
    guards the load path so concurrent first requests do not race.
    Requires: pip install "promptcue[keywords]"

    KeyBERT uses a sentence-transformer model internally — by default the same
    all-MiniLM-L6-v2 model used by PromptCueClassifier, so no additional download
    is required when promptcue[semantic] is already installed.
    """

    def __init__(self, enabled: bool = False, max_keywords: int = 8) -> None:
        self.enabled      = enabled
        self.max_keywords = max_keywords
        self._kw_model:   Any = None  # KeyBERT instance; typed as Any to avoid hard dep
        self._lock        = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """True after the KeyBERT model has been initialised."""
        return self._kw_model is not None

    def warm_up(self) -> None:
        """Pre-load KeyBERT at application startup.

        Only has effect when enabled=True. Safe to call multiple times.
        """
        if self.enabled:
            self._ensure_model()

    def extract(self, text: str) -> list[PromptCueKeyword]:
        """Return up to max_keywords keyphrases for *text*.

        Returns an empty list when disabled or when *text* is blank.
        """
        if not self.enabled or not text.strip():
            return []

        self._ensure_model()
        raw: list[tuple[str, float]] = self._kw_model.extract_keywords(
            text,
            keyphrase_ngram_range = (1, 2),
            stop_words            = 'english',
            top_n                 = self.max_keywords,
        )
        return [
            PromptCueKeyword(text=phrase, weight=round(score, 4), kind='keyphrase')
            for phrase, score in raw
        ]

    # ==============================================================================
    # Private
    # ==============================================================================

    def _ensure_model(self) -> None:
        """Initialise KeyBERT on first use. Raises a clear error for missing dep."""
        if self._kw_model is not None:
            return
        with self._lock:
            if self._kw_model is not None:
                return
            try:
                from keybert import KeyBERT
            except ImportError as exc:
                raise ImportError(
                    'Keyword extraction requires KeyBERT. '
                    'Install it with: pip install "promptcue[keywords]"'
                ) from exc
            self._kw_model = KeyBERT()
