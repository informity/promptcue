# promptcue | Main orchestration entry point for PromptCue
# Maintainer: Informity

from __future__ import annotations

import asyncio
import re

from promptcue.config import PromptCueConfig
from promptcue.constants import (
    PCUE_BASIS_SEMANTIC,
    PCUE_DEFAULT_REGISTRY,
    PCUE_HINT_STRUCTURE,
    PCUE_SCHEMA_VERSION,
)
from promptcue.core.classifier import PromptCueClassifier
from promptcue.core.decision import PromptCueDecisionEngine
from promptcue.core.registry import PromptCueRegistry
from promptcue.extraction.keywords import PromptCueKeywordExtractor
from promptcue.extraction.language import PromptCueLanguageDetector
from promptcue.extraction.linguistic import PromptCueLinguisticExtractor
from promptcue.extraction.normalization import normalize_text
from promptcue.models.schema import PromptCueQueryObject

# ==============================================================================
# Pre-classification detectors — pure regex, no model dependency
# ==============================================================================

# Openers that indicate the query is a follow-up to a previous turn.
_CONTINUATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'^\s*also[,\s]', re.IGNORECASE),
    re.compile(r'^\s*and\s+(what\s+about|also|another)\b', re.IGNORECASE),
    re.compile(r'^\s*what\s+about\s', re.IGNORECASE),
    re.compile(r'^\s*furthermore[,\s]', re.IGNORECASE),
    re.compile(r'^\s*additionally[,\s]', re.IGNORECASE),
    re.compile(r'^\s*following\s+up\b', re.IGNORECASE),
    re.compile(r'^\s*oh\s+and\b', re.IGNORECASE),
    re.compile(r'^\s*one\s+more\s+(thing|question)\b', re.IGNORECASE),
    re.compile(r'^\s*building\s+on\s+(that|this|what)', re.IGNORECASE),
    re.compile(r'^\s*to\s+follow\s+up\b', re.IGNORECASE),
    re.compile(r'^\s*going\s+back\s+to\b', re.IGNORECASE),
    re.compile(r'^\s*on\s+that\s+(note|topic|subject)\b', re.IGNORECASE),
    re.compile(r'^\s*related\s+to\s+that\b', re.IGNORECASE),
]

# Patterns indicating the caller wants a specific output structure.
_STRUCTURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'#{1,6}\s+\w', re.MULTILINE),          # Markdown headings in the query
    re.compile(r'\bformat\s+(this\s+)?as\b', re.IGNORECASE),
    re.compile(r'\boutput\s+(this\s+)?as\s+(a\s+)?table\b', re.IGNORECASE),
    re.compile(r'\bin\s+a\s+(markdown\s+)?table\b', re.IGNORECASE),
    re.compile(r'\bas\s+a\s+table\b', re.IGNORECASE),
    re.compile(r'\bformatted\s+as\b', re.IGNORECASE),
    re.compile(r'\bin\s+bullet\s+(points?|form)\b', re.IGNORECASE),
    re.compile(r'\bwith\s+(?:the\s+following\s+)?sections?:', re.IGNORECASE),
    re.compile(r'\borganiz(?:e|ed)\s+(?:it\s+)?as\b', re.IGNORECASE),
    re.compile(r'\bstructur(?:e|ed)\s+(?:it\s+)?as\b', re.IGNORECASE),
    re.compile(r'\bpresent\s+(?:this|it)\s+as\b', re.IGNORECASE),
]


def _detect_continuation(text: str) -> bool:
    """Return True when text appears to be a follow-up turn in a conversation.

    Uses leading-phrase regex patterns only — no model dependency.
    Does not change primary_query_type; purely informational for callers
    that maintain session context.
    """
    return any(pat.search(text) for pat in _CONTINUATION_PATTERNS)


def _detect_needs_structure(text: str) -> bool:
    """Return True when text contains explicit output-structure directives.

    Matches Markdown heading patterns, 'format as table', 'in bullet points',
    'with sections:', etc.  These indicate the caller has prescribed a response
    format, which downstream generators should respect.
    """
    return any(pat.search(text) for pat in _STRUCTURE_PATTERNS)


class PromptCueAnalyzer:
    """Public entry point for query understanding."""

    def __init__(self, config: PromptCueConfig | None = None) -> None:
        self.config            = config or PromptCueConfig()
        registry_path          = self.config.registry_path or PCUE_DEFAULT_REGISTRY
        self.registry          = PromptCueRegistry.from_yaml(registry_path)
        self.classifier        = PromptCueClassifier(self.registry, self.config)
        self.decision_engine   = PromptCueDecisionEngine(self.config, self.registry)
        self.language_detector    = PromptCueLanguageDetector(
            enabled = self.config.enable_language_detection,
        )
        self.linguistic_extractor = PromptCueLinguisticExtractor(
            enabled    = self.config.enable_linguistic_extraction,
            model_name = self.config.spacy_model,
        )
        self.keyword_extractor    = PromptCueKeywordExtractor(
            enabled      = self.config.enable_keyword_extraction,
            max_keywords = self.config.max_keywords,
        )

    # ==============================================================================
    # Public
    # ==============================================================================

    def warm_up(self) -> None:
        """Pre-load all optional models at application startup.

        Covers:
        - Sentence-transformer embedding model (when enable_semantic_scoring=True)
        - spaCy language model (when enable_linguistic_extraction=True)
        - KeyBERT model (when enable_keyword_extraction=True)
        - langdetect library (when enable_language_detection=True)

        Safe to call when none of the above are enabled — each component
        guards internally and becomes a no-op when disabled.
        """
        self.classifier.warm_up()
        self.linguistic_extractor.warm_up()
        self.keyword_extractor.warm_up()
        self.language_detector.warm_up()

    async def warm_up_async(self) -> None:
        """Async equivalent of warm_up() — safe to await from any async startup handler.

        Runs warm_up() in a thread-pool executor so the event loop is not
        blocked while models are loading (~10–15 s on first run, ~1–2 s
        when models are cached locally).
        """
        await asyncio.to_thread(self.warm_up)

    async def analyze_async(self, text: str) -> PromptCueQueryObject:
        """Async equivalent of analyze() — safe to await from any async context.

        Runs analyze() in a thread-pool executor so ML inference does not
        block the event loop.  Models must be loaded before calling this
        (call warm_up_async() at startup) otherwise the first request will
        pay the full model-load cost inside the executor.
        """
        return await asyncio.to_thread(self.analyze, text)

    def analyze(self, text: str) -> PromptCueQueryObject:
        """Analyze a natural-language query and return a structured PromptCueQueryObject."""
        normalized     = normalize_text(text)
        language       = self.language_detector.detect(normalized)

        # Pre-classification structural signals — pure regex, no model dependency.
        is_continuation = _detect_continuation(normalized)
        needs_structure = _detect_needs_structure(text)   # use raw text for Markdown patterns

        classification = self.classifier.classify(normalized)

        # Use the semantic threshold only when the classifier actually ran the
        # semantic path — deterministic results should be evaluated against the
        # deterministic threshold.
        top_basis = classification.candidates[0].basis if classification.candidates else None
        threshold = (
            self.config.semantic_similarity_threshold
            if top_basis == PCUE_BASIS_SEMANTIC
            else None
        )

        decision   = self.decision_engine.resolve(classification, threshold_override=threshold)
        linguistic = self.linguistic_extractor.extract(normalized)
        keywords   = self.keyword_extractor.extract(normalized)

        # Merge computed routing hints on top of YAML-derived hints from the decision engine.
        routing_hints = {**decision.routing_hints, PCUE_HINT_STRUCTURE: needs_structure}

        return PromptCueQueryObject(
            schema_version         = PCUE_SCHEMA_VERSION,
            input_text             = text,
            normalized_text        = normalized,
            language               = language,
            is_continuation        = is_continuation,
            primary_query_type     = decision.primary_label,
            classification_basis   = decision.classification_basis,
            candidate_query_types  = classification.candidates,
            confidence             = decision.confidence,
            confidence_band        = decision.confidence_band,
            ambiguity_score        = decision.ambiguity_score,
            scope                  = decision.scope,
            main_verbs             = linguistic.main_verbs,
            noun_phrases           = linguistic.noun_phrases,
            named_entities         = linguistic.named_entities,
            entities               = linguistic.entities,
            keywords               = keywords,
            routing_hints          = routing_hints,
            action_hints           = decision.action_hints,
        )
