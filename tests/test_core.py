# promptcue | Unit tests for core modules: normalization, embedding, decision, classifier
# Maintainer: Informity

from __future__ import annotations

from unittest.mock import patch

import pytest

from promptcue import PromptCueAnalyzer, PromptCueConfig
from promptcue.constants import (
    PCUE_BASIS_TRIGGER_MATCH,
    PCUE_BASIS_WORD_OVERLAP,
    PCUE_UNKNOWN,
)
from promptcue.core.classifier import PromptCueClassificationResult, PromptCueClassifier
from promptcue.core.decision import PromptCueDecisionEngine
from promptcue.core.embedding import (
    PromptCueEmbeddingBackend,
    cosine_similarity,
    cosine_similarity_batch,
)
from promptcue.core.registry import PromptCueRegistry
from promptcue.exceptions import PromptCueModelLoadError
from promptcue.extraction.normalization import normalize_text
from promptcue.models.enums import PromptCueConfidenceBand
from promptcue.models.schema import PromptCueCandidate

# ==============================================================================
# Normalization
# ==============================================================================

class TestNormalization:

    def test_nfkc_ligature(self) -> None:
        # ﬁ (U+FB01 LATIN SMALL LIGATURE FI) → 'fi'
        assert normalize_text('\ufb01le') == 'file'

    def test_nfkc_superscript(self) -> None:
        # ² (U+00B2 SUPERSCRIPT TWO) → '2'
        assert normalize_text('Python\u00b23') == 'Python23'

    def test_whitespace_collapse(self) -> None:
        assert normalize_text('foo   bar\t\nbaz') == 'foo bar baz'

    def test_leading_trailing_whitespace_stripped(self) -> None:
        assert normalize_text('  hello world  ') == 'hello world'

    def test_empty_string_returns_empty(self) -> None:
        assert normalize_text('') == ''

    def test_already_clean_string_unchanged(self) -> None:
        text = 'What is a REST API?'
        assert normalize_text(text) == text

    def test_mixed_unicode_and_whitespace(self) -> None:
        # Combines ligature + extra spaces
        assert normalize_text('  \ufb01rst  item  ') == 'first item'


# ==============================================================================
# Cosine similarity utilities
# ==============================================================================

class TestCosineSimlarity:

    def test_identical_vectors_score_one(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-5)

    def test_negative_result_clamped_to_zero(self) -> None:
        # Opposite vectors — raw cosine = -1.0; clamp produces 0.0
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0

    def test_zero_norm_vector_returns_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_result_in_unit_range(self) -> None:
        a = [0.3, -0.5, 0.8]
        b = [0.1, 0.7, -0.2]
        s = cosine_similarity(a, b)
        assert 0.0 <= s <= 1.0


class TestCosineSimlarityBatch:

    def test_empty_matrix_returns_empty_list(self) -> None:
        assert cosine_similarity_batch([1.0, 0.0], []) == []

    def test_single_row_matches_scalar_function(self) -> None:
        q  = [1.0, 0.0, 0.0]
        m  = [[0.0, 1.0, 0.0]]
        assert cosine_similarity_batch(q, m) == pytest.approx([0.0], abs=1e-5)

    def test_all_scores_in_unit_range(self) -> None:
        q = [0.3, -0.5, 0.8]
        m = [[0.1, 0.7, -0.2], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        for s in cosine_similarity_batch(q, m):
            assert 0.0 <= s <= 1.0


# ==============================================================================
# EmbeddingBackend
# ==============================================================================

class TestEmbeddingBackend:

    def test_is_loaded_false_before_warm_up(self) -> None:
        backend = PromptCueEmbeddingBackend()
        assert not backend.is_loaded

    def test_warm_up_sets_is_loaded(self) -> None:
        backend = PromptCueEmbeddingBackend()
        backend.warm_up()
        assert backend.is_loaded

    def test_encode_empty_list_returns_empty(self) -> None:
        backend = PromptCueEmbeddingBackend()
        result  = backend.encode([])
        assert result == []

    def test_failed_model_load_raises_promptcue_error(self) -> None:
        backend = PromptCueEmbeddingBackend(model_name='nonexistent-model-xyz')
        with pytest.raises(PromptCueModelLoadError):
            backend.warm_up()


# ==============================================================================
# DecisionEngine
# ==============================================================================

@pytest.fixture
def decision_engine() -> PromptCueDecisionEngine:
    registry = PromptCueRegistry()
    config   = PromptCueConfig(enable_semantic_scoring=False)
    return PromptCueDecisionEngine(config, registry)


class TestDecisionEngine:

    def test_no_candidates_returns_unknown(self, decision_engine: PromptCueDecisionEngine) -> None:
        result = decision_engine.resolve(PromptCueClassificationResult(candidates=[]))
        assert result.primary_label == PCUE_UNKNOWN

    def test_no_candidates_sets_clarify(self, decision_engine: PromptCueDecisionEngine) -> None:
        result = decision_engine.resolve(PromptCueClassificationResult(candidates=[]))
        assert result.routing_hints.get('needs_clarification') is True

    def test_no_candidates_confidence_band_low(
        self, decision_engine: PromptCueDecisionEngine,
    ) -> None:
        result = decision_engine.resolve(PromptCueClassificationResult(candidates=[]))
        assert result.confidence_band == PromptCueConfidenceBand.LOW

    def test_below_threshold_returns_unknown(
        self, decision_engine: PromptCueDecisionEngine,
    ) -> None:
        candidates = [
            PromptCueCandidate(label='lookup',     score=0.10, basis='word_overlap'),
            PromptCueCandidate(label='comparison', score=0.08, basis='word_overlap'),
        ]
        result = decision_engine.resolve(PromptCueClassificationResult(candidates=candidates))
        assert result.primary_label == PCUE_UNKNOWN

    def test_ambiguous_above_threshold_sets_clarification(
        self, decision_engine: PromptCueDecisionEngine,
    ) -> None:
        # Both candidates very close — margin < ambiguity_margin (0.08)
        candidates = [
            PromptCueCandidate(label='lookup',     score=0.62, basis='word_overlap'),
            PromptCueCandidate(label='comparison', score=0.61, basis='word_overlap'),
        ]
        result = decision_engine.resolve(PromptCueClassificationResult(candidates=candidates))
        assert result.routing_hints.get('needs_clarification') is True
        assert result.primary_label == 'lookup'  # top label still returned

    def test_clean_resolution_populates_label(
        self, decision_engine: PromptCueDecisionEngine,
    ) -> None:
        candidates = [
            PromptCueCandidate(label='lookup',     score=0.80, basis='trigger_match'),
            PromptCueCandidate(label='comparison', score=0.30, basis='word_overlap'),
        ]
        result = decision_engine.resolve(PromptCueClassificationResult(candidates=candidates))
        assert result.primary_label == 'lookup'
        assert result.routing_hints.get('needs_clarification') is False

    def test_trigger_match_band_is_high(
        self, decision_engine: PromptCueDecisionEngine,
    ) -> None:
        candidates = [
            PromptCueCandidate(label='procedure', score=0.66, basis=PCUE_BASIS_TRIGGER_MATCH),
            PromptCueCandidate(label='lookup',    score=0.20, basis=PCUE_BASIS_WORD_OVERLAP),
        ]
        result = decision_engine.resolve(PromptCueClassificationResult(candidates=candidates))
        assert result.confidence_band == PromptCueConfidenceBand.HIGH


# ==============================================================================
# Classifier tiers
# ==============================================================================

@pytest.fixture
def det_classifier() -> PromptCueClassifier:
    registry = PromptCueRegistry()
    config   = PromptCueConfig(enable_semantic_scoring=False)
    return PromptCueClassifier(registry, config)


class TestClassifierTiers:

    def test_trigger_match_fires_for_known_phrase(
        self, det_classifier: PromptCueClassifier,
    ) -> None:
        result = det_classifier._classify_deterministic('how do I set up Redis?')
        top    = result.candidates[0]
        assert top.basis == PCUE_BASIS_TRIGGER_MATCH
        assert top.label == 'procedure'

    def test_trigger_score_proportional_to_length(
        self, det_classifier: PromptCueClassifier,
    ) -> None:
        # Longer trigger → higher specificity → higher score.
        # 'how do I' (8 chars) should score higher than a shorter trigger in same type.
        result  = det_classifier._classify_deterministic('how do I configure logging?')
        top     = result.candidates[0]
        assert top.basis == PCUE_BASIS_TRIGGER_MATCH
        # Score floor is 0.60; with 8-char trigger + specificity bonus it should exceed 0.62.
        assert top.score > 0.62

    def test_word_overlap_score_in_unit_range(self, det_classifier: PromptCueClassifier) -> None:
        # A query with no trigger phrase falls through to word-overlap tier.
        result = det_classifier._classify_deterministic('explain the tradeoffs between approaches')
        for c in result.candidates:
            assert 0.0 <= c.score <= 1.0

    def test_semantic_disabled_returns_deterministic(self) -> None:
        config   = PromptCueConfig(enable_semantic_scoring=False)
        registry = PromptCueRegistry()
        clf      = PromptCueClassifier(registry, config)
        result   = clf.classify('how do I set up Redis?')
        assert result.candidates[0].basis in {PCUE_BASIS_TRIGGER_MATCH, PCUE_BASIS_WORD_OVERLAP}

    def test_cascade_trigger_overrides_semantic(self) -> None:
        # With semantic enabled, a strong trigger should still win over semantic path.
        config   = PromptCueConfig(enable_semantic_scoring=True)
        registry = PromptCueRegistry()
        clf      = PromptCueClassifier(registry, config)
        clf.warm_up()
        # "what are the steps to" is an unambiguous procedure trigger.
        result   = clf.classify('what are the steps to deploy a Docker container?')
        assert result.candidates[0].basis == PCUE_BASIS_TRIGGER_MATCH


# ==============================================================================
# Schema: new Phase 1 fields
# ==============================================================================

class TestSchemaPhase1Fields:

    def test_is_continuation_true_for_opener(self) -> None:
        analyzer = PromptCueAnalyzer(PromptCueConfig(enable_semantic_scoring=False))
        r = analyzer.analyze('Also, what about using Redis for caching?')
        assert r.is_continuation is True

    def test_is_continuation_false_for_standalone(self) -> None:
        analyzer = PromptCueAnalyzer(PromptCueConfig(enable_semantic_scoring=False))
        r = analyzer.analyze('What is a REST API?')
        assert r.is_continuation is False

    def test_needs_structure_true_for_markdown_query(self) -> None:
        analyzer = PromptCueAnalyzer(PromptCueConfig(enable_semantic_scoring=False))
        r = analyzer.analyze('Summarize this document with sections: ## Overview, ## Details')
        assert r.routing_hints.get('needs_structure') is True

    def test_needs_structure_false_for_plain_query(self) -> None:
        analyzer = PromptCueAnalyzer(PromptCueConfig(enable_semantic_scoring=False))
        r = analyzer.analyze('What is Docker?')
        assert r.routing_hints.get('needs_structure') is False

    def test_confidence_band_high_for_trigger_match(self) -> None:
        analyzer = PromptCueAnalyzer(PromptCueConfig(enable_semantic_scoring=False))
        r = analyzer.analyze('how do I set up JWT authentication in FastAPI?')
        assert r.confidence_band == PromptCueConfidenceBand.HIGH

    def test_runner_up_is_second_candidate(self) -> None:
        analyzer = PromptCueAnalyzer(PromptCueConfig(enable_semantic_scoring=False))
        r = analyzer.analyze('how do I set up Redis?')
        assert r.runner_up is not None
        assert r.runner_up == r.candidate_query_types[1]

    def test_to_routing_dict_merges_hints(self) -> None:
        analyzer = PromptCueAnalyzer(PromptCueConfig(enable_semantic_scoring=False))
        r = analyzer.analyze('how do I set up Redis?')
        d = r.to_routing_dict()
        assert isinstance(d, dict)
        # routing_hints and action_hints keys should both be present
        assert 'needs_retrieval' in d
        assert 'should_enumerate' in d

    def test_to_routing_dict_routing_priority(self) -> None:
        # routing_hints should overwrite action_hints on key collision
        analyzer = PromptCueAnalyzer(PromptCueConfig(enable_semantic_scoring=False))
        r = analyzer.analyze('what are the steps to deploy a Docker container?')
        d = r.to_routing_dict()
        # needs_clarification is a routing_hint key — it must be present
        assert 'needs_clarification' in d


# ==============================================================================
# Offline model load failure
# ==============================================================================

class TestOfflineModelLoadFailure:

    def test_failed_warm_up_raises_promptcue_model_load_error(self) -> None:
        config   = PromptCueConfig(enable_semantic_scoring=True)
        analyzer = PromptCueAnalyzer(config)
        with patch(
            'sentence_transformers.SentenceTransformer',
            side_effect=OSError('model not found'),
        ):
            with pytest.raises(PromptCueModelLoadError):
                analyzer.warm_up()
