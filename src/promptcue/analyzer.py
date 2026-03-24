# promptcue | Main orchestration entry point for PromptCue
# Maintainer: Informity

from __future__ import annotations

from promptcue.config import PromptCueConfig
from promptcue.constants import PCUE_BASIS_SEMANTIC, PCUE_DEFAULT_REGISTRY, PCUE_SCHEMA_VERSION
from promptcue.core.classifier import PromptCueClassifier
from promptcue.core.decision import PromptCueDecisionEngine
from promptcue.core.registry import PromptCueRegistry
from promptcue.extraction.keywords import PromptCueKeywordExtractor
from promptcue.extraction.language import PromptCueLanguageDetector
from promptcue.extraction.linguistic import PromptCueLinguisticExtractor
from promptcue.extraction.normalization import normalize_text
from promptcue.models.schema import PromptCueQueryObject


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

        Safe to call when none of the above are enabled — each component
        guards internally and becomes a no-op when disabled.
        """
        self.classifier.warm_up()
        self.linguistic_extractor.warm_up()
        self.keyword_extractor.warm_up()

    def analyze(self, text: str) -> PromptCueQueryObject:
        """Analyze a natural-language query and return a structured PromptCueQueryObject."""
        normalized = normalize_text(text)
        language   = self.language_detector.detect(text)

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

        return PromptCueQueryObject(
            schema_version         = PCUE_SCHEMA_VERSION,
            input_text             = text,
            normalized_text        = normalized,
            language               = language,
            primary_query_type     = decision.primary_label,
            classification_basis   = decision.classification_basis,
            candidate_query_types  = classification.candidates,
            confidence             = decision.confidence,
            ambiguity_score        = decision.ambiguity_score,
            scope                  = decision.scope,
            main_verbs             = linguistic.main_verbs,
            noun_phrases           = linguistic.noun_phrases,
            named_entities         = linguistic.named_entities,
            entities               = linguistic.entities,
            keywords               = keywords,
            routing_hints          = decision.routing_hints,
            action_hints           = decision.action_hints,
        )
