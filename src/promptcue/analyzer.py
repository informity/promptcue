# promptcue | Main orchestration entry point for PromptCue
# Maintainer: Informity

from __future__ import annotations

import asyncio
import re

from promptcue.config import PromptCueConfig
from promptcue.constants import (
    PCUE_DEFAULT_REGISTRY,
    PCUE_SCHEMA_VERSION,
)
from promptcue.core.classifier import PromptCueClassifier
from promptcue.core.decision import PromptCueDecisionEngine
from promptcue.core.registry import PromptCueRegistry
from promptcue.extraction.keywords import PromptCueKeywordExtractor
from promptcue.extraction.language import PromptCueLanguageDetector
from promptcue.extraction.linguistic import PromptCueLinguisticExtractor
from promptcue.extraction.normalization import normalize_text
from promptcue.models.enums import PromptCueBasis, PromptCueRoutingHint
from promptcue.models.schema import (
    PromptCueConfidenceMeta,
    PromptCueExplanations,
    PromptCueQueryObject,
    PromptCueSemanticHints,
)

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
_MULTI_ITEM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r'\b(across all|all (?:documents|files|records)|multiple (?:documents|files|records))\b',
        re.IGNORECASE,
    ),
    re.compile(r'\b(documents|files|records)\b', re.IGNORECASE),
]
_COMPARISON_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r'\b(compare|contrast|versus|vs\.?|trade[-\s]*offs?|pros?\s+and\s+cons?)\b',
        re.IGNORECASE,
    ),
]
_ENUMERATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r'\b(list|enumerate|step[-\s]*by[-\s]*step|top\s+\d+|bullet(?:s| points?)?)\b',
        re.IGNORECASE,
    ),
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


# Patterns indicating the query has a temporal scope — references a specific time
# period, a year-over-year comparison, or a temporal aggregation.  Pure regex,
# no model dependency; runs before classification.
#
# Scope: generic signals any time-aware application can act on.
# Corpus-specific interpretations (e.g. aggregate_by_period, group_by='year')
# are the caller's responsibility and must stay in the consuming application.
_TEMPORAL_SCOPE_PATTERNS: list[re.Pattern[str]] = [
    # Specific 4-digit year reference (1900–2099)
    re.compile(r'\b(?:19|20)\d{2}\b'),
    # Year-over-year and multi-year aggregation phrases
    re.compile(r'\byear[-\s]*(?:by|over)[-\s]*year\b',   re.IGNORECASE),
    re.compile(r'\bcross[-\s]*year\b',                    re.IGNORECASE),
    re.compile(r'\bby\s+year\b',                          re.IGNORECASE),
    re.compile(r'\byears?\s+covered\b',                   re.IGNORECASE),
    re.compile(r'\byear[-\s]*to[-\s]*date\b',             re.IGNORECASE),
    re.compile(r'\bYTD\b'),
    # Duration phrases: "over/in/for the last/past N years"
    re.compile(
        r'\b(?:over|in|for)\s+the\s+(?:last|past)\s+\d+\s+years?\b', re.IGNORECASE,
    ),
    # Explicit year ranges: "from 2020 to 2023", "between 2018 and 2022"
    re.compile(r'\bfrom\s+(?:19|20)\d{2}\s+to\s+(?:19|20)\d{2}\b',        re.IGNORECASE),
    re.compile(r'\bbetween\s+(?:19|20)\d{2}\s+and\s+(?:19|20)\d{2}\b',    re.IGNORECASE),
    # "since 2020", "starting from 2019"
    re.compile(r'\bsince\s+(?:19|20)\d{2}\b',         re.IGNORECASE),
    re.compile(r'\bstarting\s+(?:from\s+)?(?:19|20)\d{2}\b', re.IGNORECASE),
    # Periodic trend phrases
    re.compile(r'\b(?:quarterly|annual|monthly)\s+trend\b', re.IGNORECASE),
    re.compile(r'\bover\s+time\b',                          re.IGNORECASE),
    # Quarter references: "Q1 2023", "2023 Q4"
    re.compile(r'\bQ[1-4]\s+(?:19|20)\d{2}\b'),
    re.compile(r'\b(?:19|20)\d{2}\s+Q[1-4]\b'),
]


def _detect_mentions_time(text: str) -> bool:
    """Return True when text references a specific time period or temporal aggregation.

    Covers year references (2020, 2021...), year-over-year phrases, multi-year
    duration phrases ("over the last 3 years"), year ranges, and periodic trend
    phrases.  Pure regex — no model dependency.

    Populates the generic semantic hint `mentions_time`.
    """
    return any(pat.search(text) for pat in _TEMPORAL_SCOPE_PATTERNS)


_MULTI_PERIOD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'\byear[-\s]*(?:by|over)[-\s]*year\b', re.IGNORECASE),
    re.compile(r'\bcross[-\s]*year\b', re.IGNORECASE),
    re.compile(r'\bby\s+year\b', re.IGNORECASE),
    re.compile(r'\bover\s+time\b', re.IGNORECASE),
    re.compile(r'\b(?:quarterly|annual|monthly)\s+trend\b', re.IGNORECASE),
    re.compile(r'\b(?:over|in|for)\s+the\s+(?:last|past)\s+\d+\s+years?\b', re.IGNORECASE),
    re.compile(r'\bfrom\s+(?:19|20)\d{2}\s+to\s+(?:19|20)\d{2}\b', re.IGNORECASE),
    re.compile(r'\bbetween\s+(?:19|20)\d{2}\s+and\s+(?:19|20)\d{2}\b', re.IGNORECASE),
]


def _detect_requires_multi_period_analysis(text: str) -> bool:
    """Return True for prompts that explicitly require analysis across periods."""
    if any(pat.search(text) for pat in _MULTI_PERIOD_PATTERNS):
        return True
    years = re.findall(r'\b(?:19|20)\d{2}\b', text)
    return len(set(years)) >= 2


def _detect_mentions_multiple_items(text: str) -> bool:
    return any(pat.search(text) for pat in _MULTI_ITEM_PATTERNS)


def _detect_requests_comparison(text: str) -> bool:
    return any(pat.search(text) for pat in _COMPARISON_PATTERNS)


def _detect_requests_enumeration(text: str) -> bool:
    return any(pat.search(text) for pat in _ENUMERATION_PATTERNS)


def _extract_evidence_tokens(text: str, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in re.findall(r'[a-z0-9][a-z0-9_-]{2,}', text.casefold()):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


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
        is_continuation  = _detect_continuation(normalized)
        needs_structure  = _detect_needs_structure(text)   # use raw text for Markdown patterns
        mentions_time = _detect_mentions_time(normalized)
        requires_multi_period_analysis = _detect_requires_multi_period_analysis(normalized)
        mentions_multiple_items = _detect_mentions_multiple_items(normalized)
        requests_comparison = _detect_requests_comparison(normalized)
        requests_enumeration = _detect_requests_enumeration(normalized)

        classification = self.classifier.classify(normalized)

        # Use the semantic threshold only when the classifier actually ran the
        # semantic path — deterministic results should be evaluated against the
        # deterministic threshold.
        top_basis = classification.candidates[0].basis if classification.candidates else None
        threshold = (
            self.config.semantic_similarity_threshold
            if top_basis == PromptCueBasis.SEMANTIC
            else None
        )

        decision   = self.decision_engine.resolve(classification, threshold_override=threshold)
        linguistic = self.linguistic_extractor.extract(normalized)
        keywords   = self.keyword_extractor.extract(normalized)

        # Merge computed routing hints on top of YAML-derived hints from the decision engine.
        routing_hints = {
            **decision.routing_hints,
            PromptCueRoutingHint.NEEDS_STRUCTURE: needs_structure,
        }

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
            confidence_meta        = PromptCueConfidenceMeta(
                type_confidence_margin=decision.type_confidence_margin,
                scope_confidence=decision.scope_confidence,
                scope_confidence_margin=decision.scope_confidence_margin,
            ),
            scope                  = decision.scope,
            main_verbs             = linguistic.main_verbs,
            noun_phrases           = linguistic.noun_phrases,
            named_entities         = linguistic.named_entities,
            entities               = linguistic.entities,
            keywords               = keywords,
            routing_hints          = routing_hints,
            action_hints           = decision.action_hints,
            semantic_hints         = PromptCueSemanticHints(
                mentions_multiple_items=mentions_multiple_items,
                requests_comparison=requests_comparison,
                requests_enumeration=requests_enumeration,
                requests_structure=needs_structure,
                mentions_time=mentions_time,
                requires_multi_period_analysis=requires_multi_period_analysis,
            ),
            explanations           = PromptCueExplanations(
                decision_notes=decision.decision_notes,
                evidence_tokens=_extract_evidence_tokens(normalized),
            ),
        )
