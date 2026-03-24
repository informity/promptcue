# Semantic scoring tests — requires the 'semantic' extra (sentence-transformers).
# Tests are skipped automatically if sentence-transformers is not installed.
# Run with: pytest tests/test_semantic.py -v

import pytest

sentence_transformers = pytest.importorskip(
    'sentence_transformers',
    reason='sentence-transformers not installed; skipping (pip install "promptcue[semantic]")',
)

from promptcue import PromptCueAnalyzer, PromptCueConfig  # noqa: E402

# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope='module')
def semantic_analyzer() -> PromptCueAnalyzer:
    """Single analyzer instance shared across all semantic tests — model loads once."""
    analyzer = PromptCueAnalyzer(PromptCueConfig(enable_semantic_scoring=True))
    analyzer.warm_up()
    return analyzer


# ==============================================================================
# Tests
# ==============================================================================

def test_semantic_analyzer_returns_object(semantic_analyzer: PromptCueAnalyzer) -> None:
    result = semantic_analyzer.analyze('Compare Aurora and OpenSearch for RAG on AWS')
    assert result.input_text
    assert result.primary_query_type
    assert 0.0 <= result.confidence <= 1.0


def test_semantic_candidates_use_semantic_basis(semantic_analyzer: PromptCueAnalyzer) -> None:
    # "compare" is an explicit trigger → trigger_confident fires and the deterministic
    # path is returned.  Verify the top candidate correctly identifies comparison.
    result = semantic_analyzer.analyze('Compare Aurora and OpenSearch for RAG on AWS')
    assert result.primary_query_type == 'comparison'
    assert result.classification_basis in ('trigger_match', 'semantic_similarity')


def test_semantic_scores_comparison_query(semantic_analyzer: PromptCueAnalyzer) -> None:
    result = semantic_analyzer.analyze('Compare Aurora and OpenSearch for RAG on AWS')
    assert result.primary_query_type == 'comparison'
    assert result.confidence > 0.2


def test_semantic_scores_lookup_query(semantic_analyzer: PromptCueAnalyzer) -> None:
    result = semantic_analyzer.analyze('What is the default retention period for CloudWatch Logs?')
    assert result.primary_query_type == 'lookup'


def test_semantic_scores_troubleshooting_query(semantic_analyzer: PromptCueAnalyzer) -> None:
    result = semantic_analyzer.analyze('Our Lambda function keeps timing out — how do I fix it?')
    assert result.primary_query_type == 'troubleshooting'


def test_semantic_scores_coverage_query(semantic_analyzer: PromptCueAnalyzer) -> None:
    result = semantic_analyzer.analyze('Give me a broad overview of AWS networking services')
    assert result.primary_query_type == 'coverage'


def test_semantic_scores_recommendation_query(semantic_analyzer: PromptCueAnalyzer) -> None:
    result = semantic_analyzer.analyze('Should we use DynamoDB or RDS for a high-read catalog?')
    assert result.primary_query_type == 'recommendation'


def test_semantic_routing_hints_populated(semantic_analyzer: PromptCueAnalyzer) -> None:
    result = semantic_analyzer.analyze('Compare Aurora and OpenSearch for RAG on AWS')
    assert 'needs_retrieval'    in result.routing_hints
    assert 'needs_reasoning'    in result.routing_hints
    assert 'needs_current_info' in result.routing_hints


def test_warm_up_pre_loads_model(semantic_analyzer: PromptCueAnalyzer) -> None:
    assert semantic_analyzer.classifier.embedding_backend.is_loaded


def test_example_cache_populated_after_warm_up(semantic_analyzer: PromptCueAnalyzer) -> None:
    # Cache should have one entry per registered query type — update when types are added.
    registry_size = len(semantic_analyzer.registry.definitions)
    assert len(semantic_analyzer.classifier._example_cache) == registry_size
