# promptcue | Unit tests for core modules: normalization, embedding, decision, classifier
# Maintainer: Informity

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from promptcue import PromptCueAnalyzer, PromptCueConfig
from promptcue.analyzer import _detect_mentions_time, _detect_requires_multi_period_analysis
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

class TestCosineSimilarity:

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


class TestCosineSimilarityBatch:

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
        pytest.importorskip('sentence_transformers')
        backend = PromptCueEmbeddingBackend()
        backend.warm_up()
        assert backend.is_loaded

    def test_encode_empty_list_returns_empty(self) -> None:
        # No model needed — empty input short-circuits before _ensure_model().
        backend = PromptCueEmbeddingBackend()
        result  = backend.encode([])
        assert result == []

    def test_failed_model_load_raises_promptcue_error(self) -> None:
        # Inject a fake sentence_transformers whose constructor raises OSError so
        # the test runs without the real package installed.
        mock_st = MagicMock()
        mock_st.SentenceTransformer.side_effect = OSError('model not found')
        backend = PromptCueEmbeddingBackend(model_name='nonexistent-model-xyz')
        with (
            patch.dict('sys.modules', {'sentence_transformers': mock_st}),
            pytest.raises(PromptCueModelLoadError),
        ):
            backend.warm_up()

    def test_encode_passes_show_progress_bar_false_by_default(self) -> None:
        backend = PromptCueEmbeddingBackend()
        backend._model = MagicMock()
        backend._model.encode.return_value.tolist.return_value = [[0.0, 1.0]]

        backend.encode(['hello'])

        backend._model.encode.assert_called_once_with(
            ['hello'],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def test_encode_passes_show_progress_bar_true_when_enabled(self) -> None:
        backend = PromptCueEmbeddingBackend(show_progress_bar=True)
        backend._model = MagicMock()
        backend._model.encode.return_value.tolist.return_value = [[0.0, 1.0]]

        backend.encode(['hello'])

        backend._model.encode.assert_called_once_with(
            ['hello'],
            convert_to_numpy=True,
            show_progress_bar=True,
        )


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

    def test_conversation_summary_trigger_match(self, det_classifier: PromptCueClassifier) -> None:
        result = det_classifier._classify_deterministic('summarize our chat so far')
        top = result.candidates[0]
        assert top.basis == PCUE_BASIS_TRIGGER_MATCH
        assert top.label == 'conversation_summary'

    def test_semantic_disabled_returns_deterministic(self) -> None:
        config   = PromptCueConfig(enable_semantic_scoring=False)
        registry = PromptCueRegistry()
        clf      = PromptCueClassifier(registry, config)
        result   = clf.classify('how do I set up Redis?')
        assert result.candidates[0].basis in {PCUE_BASIS_TRIGGER_MATCH, PCUE_BASIS_WORD_OVERLAP}

    def test_cascade_trigger_overrides_semantic(self) -> None:
        pytest.importorskip('sentence_transformers')
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
        # Inject a fake sentence_transformers whose constructor raises OSError so
        # the test runs without the real package installed.
        mock_st = MagicMock()
        mock_st.SentenceTransformer.side_effect = OSError('model not found')
        config   = PromptCueConfig(enable_semantic_scoring=True)
        analyzer = PromptCueAnalyzer(config)
        with (
            patch.dict('sys.modules', {'sentence_transformers': mock_st}),
            pytest.raises(PromptCueModelLoadError),
        ):
            analyzer.warm_up()


# ==============================================================================
# Injectable embed_fn
# ==============================================================================

class TestInjectableEmbedFn:
    """Tests for the hosted (embed_fn) operating mode.

    All tests in this class run without sentence-transformers installed — the
    embed_fn replaces the model entirely.  A simple deterministic stub is used
    so the semantic path still produces consistent vectors.
    """

    # Dimension must match what the registry examples would produce — since we
    # use a stub that returns a fixed-length vector, we just need it to be
    # non-zero so cosine similarity math works.
    _DIM = 8

    @staticmethod
    def _make_embed_fn(vector: list[float] | None = None) -> Callable[[str], list[float]]:
        """Return an embed_fn stub that always returns the same vector."""
        v = vector or ([1.0] + [0.0] * (TestInjectableEmbedFn._DIM - 1))
        return lambda _text: v

    def test_embed_fn_flag_forces_semantic_scoring(self) -> None:
        # Even without sentence-transformers, enable_semantic_scoring must be
        # True when embed_fn is provided.
        config = PromptCueConfig(embed_fn=self._make_embed_fn())
        assert config.enable_semantic_scoring is True

    def test_backend_is_loaded_immediately(self) -> None:
        # is_loaded must be True without calling warm_up() first.
        backend = PromptCueEmbeddingBackend(embed_fn=self._make_embed_fn())
        assert backend.is_loaded is True

    def test_warm_up_is_noop_with_embed_fn(self) -> None:
        # warm_up() must not raise and must not attempt to load a model.
        backend = PromptCueEmbeddingBackend(embed_fn=self._make_embed_fn())
        backend.warm_up()   # should not raise
        assert backend._model is None  # internal model was never loaded

    def test_encode_delegates_to_embed_fn(self) -> None:
        expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        backend  = PromptCueEmbeddingBackend(embed_fn=self._make_embed_fn(expected))
        result   = backend.encode(['hello', 'world'])
        assert result == [expected, expected]

    def test_encode_empty_returns_empty_without_calling_fn(self) -> None:
        call_count = 0

        def counting_fn(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            return [0.0] * self._DIM

        backend = PromptCueEmbeddingBackend(embed_fn=counting_fn)
        result  = backend.encode([])
        assert result == []
        assert call_count == 0

    def test_analyzer_accepts_embed_fn_in_config(self) -> None:
        # End-to-end: PromptCueAnalyzer runs without loading sentence-transformers.
        # The stub returns a constant vector so all types score equally; the
        # deterministic trigger path will determine the actual label.  We just
        # verify no exception is raised and a PromptCueQueryObject is returned.
        config = PromptCueConfig(embed_fn=self._make_embed_fn())
        analyzer = PromptCueAnalyzer(config)
        result = analyzer.analyze('how do I set up Redis caching?')
        assert result.primary_query_type is not None

    def test_analyzer_warm_up_noop_with_embed_fn(self) -> None:
        # warm_up() on a fully-configured analyzer with embed_fn must not raise.
        config   = PromptCueConfig(embed_fn=self._make_embed_fn())
        analyzer = PromptCueAnalyzer(config)
        analyzer.warm_up()   # should not raise


# ==============================================================================
# Temporal scope detection
# ==============================================================================

class TestTemporalScope:
    """Tests for temporal detection and semantic temporal hints."""

    # -------------------------------------------------------------------
    # Direct function tests — True cases
    # -------------------------------------------------------------------

    def test_year_reference_fires(self) -> None:
        assert _detect_mentions_time('What changed in 2023?') is True
        assert _detect_requires_multi_period_analysis('What changed in 2023?') is False

    def test_year_range_fires(self) -> None:
        assert _detect_mentions_time('Compare performance from 2020 to 2023') is True
        assert (
            _detect_requires_multi_period_analysis('Compare performance from 2020 to 2023')
            is True
        )

    def test_between_years_fires(self) -> None:
        assert _detect_mentions_time('Results between 2019 and 2022') is True
        assert _detect_requires_multi_period_analysis('Results between 2019 and 2022') is True

    def test_since_year_fires(self) -> None:
        assert _detect_mentions_time('What happened since 2021?') is True

    def test_year_over_year_fires(self) -> None:
        assert _detect_mentions_time('Show year-over-year growth') is True
        assert _detect_requires_multi_period_analysis('Show year-over-year growth') is True

    def test_year_by_year_fires(self) -> None:
        assert _detect_mentions_time('Break down findings year by year') is True
        assert _detect_requires_multi_period_analysis('Break down findings year by year') is True

    def test_cross_year_fires(self) -> None:
        assert _detect_mentions_time('Cross-year analysis of incidents') is True
        assert _detect_requires_multi_period_analysis('Cross-year analysis of incidents') is True

    def test_by_year_fires(self) -> None:
        assert _detect_mentions_time('Group results by year') is True
        assert _detect_requires_multi_period_analysis('Group results by year') is True

    def test_year_to_date_fires(self) -> None:
        assert _detect_mentions_time('Show year-to-date revenue') is True

    def test_ytd_fires(self) -> None:
        assert _detect_mentions_time('What is the YTD total?') is True

    def test_last_n_years_fires(self) -> None:
        assert _detect_mentions_time('Over the last 3 years, what changed?') is True
        assert (
            _detect_requires_multi_period_analysis('Over the last 3 years, what changed?')
            is True
        )

    def test_past_n_years_fires(self) -> None:
        assert _detect_mentions_time('In the past 5 years of data') is True
        assert _detect_requires_multi_period_analysis('In the past 5 years of data') is True

    def test_quarterly_trend_fires(self) -> None:
        assert _detect_mentions_time('Show the quarterly trend for incidents') is True
        assert (
            _detect_requires_multi_period_analysis('Show the quarterly trend for incidents')
            is True
        )

    def test_annual_trend_fires(self) -> None:
        assert _detect_mentions_time('Describe the annual trend') is True

    def test_over_time_fires(self) -> None:
        assert _detect_mentions_time('How does performance degrade over time?') is True
        assert (
            _detect_requires_multi_period_analysis('How does performance degrade over time?')
            is True
        )

    def test_quarter_ref_fires(self) -> None:
        assert _detect_mentions_time('Q3 2022 summary') is True

    # -------------------------------------------------------------------
    # Direct function tests — False cases (no temporal cues)
    # -------------------------------------------------------------------

    def test_plain_query_no_fire(self) -> None:
        assert _detect_mentions_time('How do I configure Redis caching?') is False

    def test_comparison_no_year_no_fire(self) -> None:
        query = 'Compare Aurora and RDS for a high-read workload'
        assert _detect_mentions_time(query) is False
        assert (
            _detect_requires_multi_period_analysis(query)
            is False
        )

    def test_procedure_query_no_fire(self) -> None:
        assert _detect_mentions_time('How do I set up JWT authentication in FastAPI?') is False

    def test_version_number_no_fire(self) -> None:
        # "2.0", "3.11", "v1.5" — version strings must not trigger year detection
        assert _detect_mentions_time('What is new in Python 3.11?') is False
        assert _detect_mentions_time('Upgrade to HTTP/2.0') is False

    # -------------------------------------------------------------------
    # End-to-end: semantic temporal hints present in PromptCueQueryObject
    # -------------------------------------------------------------------

    def test_semantic_hints_true_for_year_query(self) -> None:
        analyzer = PromptCueAnalyzer(PromptCueConfig(enable_semantic_scoring=False))
        r = analyzer.analyze('What changed in 2023 compared to 2022?')
        assert r.semantic_hints.mentions_time is True
        assert r.semantic_hints.requires_multi_period_analysis is True

    def test_semantic_hints_false_for_plain_query(self) -> None:
        analyzer = PromptCueAnalyzer(PromptCueConfig(enable_semantic_scoring=False))
        r = analyzer.analyze('How do I set up Redis caching?')
        assert r.semantic_hints.mentions_time is False
        assert r.semantic_hints.requires_multi_period_analysis is False

    def test_semantic_hints_present_on_all_queries(self) -> None:
        analyzer = PromptCueAnalyzer(PromptCueConfig(enable_semantic_scoring=False))
        r = analyzer.analyze('What is a REST API?')
        assert hasattr(r, 'semantic_hints')
