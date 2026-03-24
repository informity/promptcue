# promptcue | Runtime configuration for PromptCue
# Maintainer: Informity

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _semantic_available() -> bool:
    """Return True when sentence-transformers is importable (i.e. installed)."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _default_model_cache_dir() -> Path | None:
    """Return the model cache directory from PROMPTCUE_MODEL_CACHE env var, or None."""
    env = os.environ.get('PROMPTCUE_MODEL_CACHE')
    return Path(env) if env else None


class PromptCueConfig(BaseModel):
    """Top-level runtime configuration for PromptCueAnalyzer."""

    # ==============================================================================
    # Registry
    # ==============================================================================
    registry_path:               Path | None = Field(default=None)

    # ==============================================================================
    # Model cache
    # Controls where sentence-transformers (and HuggingFace Hub) looks for cached
    # models.  When set, passed as cache_folder to SentenceTransformer().  Falls
    # back to PROMPTCUE_MODEL_CACHE env var if not set in config.  When neither is
    # set, HuggingFace's default cache (~/.cache/huggingface/) is used.
    #
    # Production deployments must pre-position models before starting the service
    # and call warm_up() at startup.  PromptCue does not degrade to deterministic-
    # only mode on model load failure — it raises PromptCueModelLoadError instead.
    #
    # Deployment patterns:
    #   Local dev       : leave unset — HuggingFace downloads on first use
    #   EC2 / EBS       : HF_HOME=/opt/models  or  model_cache_dir=Path('/opt/models')
    #   Lambda container: bake model into image; set HF_HOME=/app/models at build time
    #   Lambda EFS      : model_cache_dir=Path('/mnt/models')
    #   macOS desktop   : model_cache_dir=Path('~/Library/Application Support/<app>/models')
    # ==============================================================================
    model_cache_dir:             Path | None = Field(default_factory=_default_model_cache_dir)

    # ==============================================================================
    # Classification thresholds
    # Deterministic path: scores are 0.10 / 0.60–0.85 — threshold 0.55 works.
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
    # Confidence band thresholds — used by the decision engine to populate
    # confidence_band on every result.  Scores at or above confidence_high_threshold
    # map to 'high'; scores between the two thresholds map to 'medium'; below maps
    # to 'low'.  Trigger-match results always carry 'high' regardless of score, as
    # the trigger itself is an explicit, unambiguous intent signal.
    confidence_high_threshold:     float = Field(default=0.65, ge=0.0, le=1.0)
    confidence_medium_threshold:   float = Field(default=0.35, ge=0.0, le=1.0)

    # ==============================================================================
    # Semantic scoring — cascade classifier (semantic-first)
    # Enabled automatically when sentence-transformers is installed.
    # Semantic path always runs when enabled; it is the primary classification
    # signal.  The deterministic pass supplies trigger-match corrections: when a
    # trigger fires at >= trigger_fallback_threshold AND the top-two deterministic
    # scores differ by >= ambiguity_margin, the trigger result overrides semantic.
    # Model is pre-downloaded to model_cache_dir (or HuggingFace default cache).
    # ==============================================================================
    enable_semantic_scoring:    bool  = Field(default_factory=_semantic_available)
    embedding_model:            str   = Field(default='all-MiniLM-L6-v2')
    trigger_fallback_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    # Penalty subtracted from each type's semantic score per the max cosine
    # similarity with that type's negative examples.  0.0 disables the penalty.
    # Derived from informity-ai's production classifier (_NEGATIVE_PENALTY_WEIGHT).
    negative_penalty_weight:    float = Field(default=0.15, ge=0.0, le=1.0)

    # ==============================================================================
    # Language detection
    # Requires: pip install "promptcue[detection]"
    # Returns BCP-47 language code (e.g. 'en', 'fr') in PromptCueQueryObject.language.
    # ==============================================================================
    enable_language_detection:   bool = Field(default=False)

    # ==============================================================================
    # Enrichment
    # spaCy model is installed via: python -m spacy download en_core_web_sm
    # KeyBERT reuses the sentence-transformers embedding model.
    # ==============================================================================
    enable_linguistic_extraction: bool = Field(default=False)
    enable_keyword_extraction:    bool = Field(default=False)
    max_keywords:                 int  = Field(default=8, ge=1, le=100)
    spacy_model:                  str  = Field(default='en_core_web_sm')

    # ==============================================================================
    # Named calibration presets
    # ==============================================================================

    @classmethod
    def strict(cls) -> 'PromptCueConfig':
        """High-precision preset: tighter thresholds, wider ambiguity gate.

        More queries fall through to 'unknown'; fewer borderline queries are
        returned with a false-confident label.  Use when incorrect classifications
        are more costly than abstentions (e.g. automated routing without human review).
        """
        return cls(
            similarity_threshold          = 0.70,
            semantic_similarity_threshold = 0.35,
            ambiguity_margin              = 0.12,
            confidence_high_threshold     = 0.75,
            confidence_medium_threshold   = 0.50,
        )

    @classmethod
    def balanced(cls) -> 'PromptCueConfig':
        """Default balanced preset — same thresholds as the plain constructor.

        Named reference for configuration documentation and comparison tests.
        """
        return cls()

    @classmethod
    def recall_heavy(cls) -> 'PromptCueConfig':
        """High-recall preset: looser thresholds, narrower ambiguity gate.

        Fewer queries return 'unknown'; more borderline queries are classified.
        Use when coverage matters more than precision (e.g. analytics, logging,
        non-critical routing where a wrong label is tolerable).
        """
        return cls(
            similarity_threshold          = 0.35,
            semantic_similarity_threshold = 0.10,
            ambiguity_margin              = 0.04,
            confidence_high_threshold     = 0.55,
            confidence_medium_threshold   = 0.25,
        )
