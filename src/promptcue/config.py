# promptcue | Runtime configuration for PromptCue
# Maintainer: Informity

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


def _semantic_available() -> bool:
    """Return True when sentence-transformers is importable (i.e. installed)."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class PromptCueConfig(BaseModel):
    """Top-level runtime configuration for PromptCueAnalyzer."""

    # ==============================================================================
    # Registry
    # ==============================================================================
    registry_path:               Path | None = Field(default=None)

    # ==============================================================================
    # Classification thresholds
    # Deterministic path: scores are 0.10 / 0.60–0.85 / 0.90 — threshold 0.55 works.
    # Semantic path: cosine similarity typically ranges 0.20–0.80 for this model
    #   and example set — a lower threshold is required.
    # Cascade: deterministic result is kept when its top score reaches
    #   semantic_fallback_threshold AND the margin between top-2 exceeds
    #   ambiguity_margin.  Otherwise the semantic path runs as a fallback.
    # ==============================================================================
    similarity_threshold:          float = Field(default=0.55, ge=0.0, le=1.0)
    semantic_similarity_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    ambiguity_margin:              float = Field(default=0.08, ge=0.0, le=1.0)
    semantic_fallback_threshold:   float = Field(default=0.75, ge=0.0, le=1.0)

    # ==============================================================================
    # Semantic scoring — cascade classifier
    # Enabled automatically when sentence-transformers is installed.
    # The deterministic pass runs first; semantic is invoked only when the
    # deterministic confidence is below semantic_fallback_threshold or the
    # top-two candidates are too close (< ambiguity_margin apart).
    # Downloads model from HuggingFace Hub on first use (~90 MB for the default).
    # Model is cached locally after first download.
    # ==============================================================================
    enable_semantic_scoring: bool = Field(default_factory=_semantic_available)
    embedding_model:         str  = Field(default='all-MiniLM-L6-v2')

    # ==============================================================================
    # Language detection
    # Requires: pip install "promptcue[detection]"
    # Returns BCP-47 language code (e.g. 'en', 'fr') in PromptCueQueryObject.language.
    # ==============================================================================
    enable_language_detection:   bool = Field(default=False)

    # ==============================================================================
    # Enrichment
    # Downloads spaCy model on first use when linguistic extraction is enabled.
    # Model is cached locally after first download.
    # ==============================================================================
    enable_linguistic_extraction: bool = Field(default=False)
    enable_keyword_extraction:    bool = Field(default=False)
    max_keywords:                 int  = Field(default=8, ge=1, le=100)
    spacy_model:                  str  = Field(default='en_core_web_sm')
