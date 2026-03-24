# promptcue | spaCy-based linguistic feature extractor
# Maintainer: Informity

from __future__ import annotations

import threading
from typing import Any

from promptcue.models.schema import PromptCueEntity, PromptCueLinguistics


class PromptCueLinguisticExtractor:
    """Extracts linguistic features from a query using spaCy.

    The spaCy model is loaded lazily on first extract() call.  A threading.Lock
    guards the load path so concurrent first requests do not race.
    Requires:
        pip install "promptcue[linguistic]"
        python -m spacy download en_core_web_sm
    """

    def __init__(self, enabled: bool = False, model_name: str = 'en_core_web_sm') -> None:
        self.enabled    = enabled
        self.model_name = model_name
        self._nlp: Any  = None  # spacy.Language instance; typed as Any to avoid hard dep
        self._lock      = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """True after the spaCy model has been loaded into memory."""
        return self._nlp is not None

    def warm_up(self) -> None:
        """Pre-load the spaCy model at application startup.

        Only has effect when enabled=True. Safe to call multiple times.
        """
        if self.enabled:
            self._ensure_model()

    def extract(self, text: str) -> PromptCueLinguistics:
        """Return linguistic features for *text*.

        Returns an empty PromptCueLinguistics when disabled or when *text* is blank.
        """
        if not self.enabled or not text.strip():
            return PromptCueLinguistics()

        self._ensure_model()
        doc = self._nlp(text)

        # Root verbs only — the grammatical head of the main clause.
        main_verbs = [
            token.lemma_
            for token in doc
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB'
        ]

        # spaCy noun chunks are base noun phrases (no nested NPs).
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]

        # Structured entities: surface text + spaCy entity type label (ORG, GPE, PRODUCT, etc.)
        entities = [
            PromptCueEntity(text=ent.text, entity_type=ent.label_)
            for ent in doc.ents
        ]

        return PromptCueLinguistics(
            main_verbs     = main_verbs,
            noun_phrases   = noun_phrases,
            named_entities = [e.text for e in entities],  # plain text, backward compat
            entities       = entities,
        )

    # ==============================================================================
    # Private
    # ==============================================================================

    def _ensure_model(self) -> None:
        """Load the spaCy model on first use. Raises clear errors for missing deps."""
        if self._nlp is not None:
            return
        with self._lock:
            if self._nlp is not None:
                return
            try:
                import spacy
            except ImportError as exc:
                raise ImportError(
                    'Linguistic extraction requires spaCy. '
                    'Install it with: pip install "promptcue[linguistic]"'
                ) from exc
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError as exc:
                raise OSError(
                    f'spaCy model "{self.model_name}" is not installed. '
                    f'Download it with: python -m spacy download {self.model_name}'
                ) from exc
