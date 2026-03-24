import pytest

from promptcue import PromptCueAnalyzer
from promptcue.constants import (
    PCUE_BASIS_BELOW_THRESHOLD,
    PCUE_BASIS_FALLBACK,
    PCUE_BASIS_LABEL_MATCH,
    PCUE_BASIS_SEMANTIC,
    PCUE_BASIS_TRIGGER_MATCH,
    PCUE_BASIS_WORD_OVERLAP,
    PCUE_SCHEMA_VERSION,
)

_KNOWN_TYPES = {
    'coverage', 'lookup', 'comparison', 'recommendation', 'troubleshooting',
    'procedure', 'analysis', 'update', 'summarization', 'generation',
    'validation', 'chitchat', 'unknown',
}
# Cascade classifier may produce deterministic or semantic basis depending on
# which path fires — both are valid for these general tests.
_KNOWN_BASES = {
    PCUE_BASIS_LABEL_MATCH, PCUE_BASIS_TRIGGER_MATCH,
    PCUE_BASIS_WORD_OVERLAP, PCUE_BASIS_FALLBACK,
    PCUE_BASIS_BELOW_THRESHOLD, PCUE_BASIS_SEMANTIC,
}


def test_analyzer_returns_object() -> None:
    analyzer = PromptCueAnalyzer()
    result   = analyzer.analyze('compare aurora and opensearch for rag')

    assert result.input_text
    assert result.primary_query_type in _KNOWN_TYPES
    assert 0.0 <= result.confidence <= 1.0


def test_schema_version_present() -> None:
    result = PromptCueAnalyzer().analyze('compare aurora and opensearch for rag')
    assert result.schema_version == PCUE_SCHEMA_VERSION


def test_classification_basis_valid() -> None:
    result = PromptCueAnalyzer().analyze('compare aurora and opensearch for rag')
    assert result.classification_basis in _KNOWN_BASES


def test_candidates_carry_basis() -> None:
    result = PromptCueAnalyzer().analyze('compare aurora and opensearch for rag')
    assert all(c.basis in _KNOWN_BASES for c in result.candidate_query_types)


def test_routing_hints_keys_present() -> None:
    result = PromptCueAnalyzer().analyze('compare aurora and opensearch for rag')
    assert 'needs_clarification' in result.routing_hints
    assert 'needs_retrieval'     in result.routing_hints
    assert 'needs_reasoning'     in result.routing_hints
    assert 'needs_current_info'  in result.routing_hints


# ==============================================================================
# Async interface
# ==============================================================================

@pytest.mark.asyncio
async def test_analyze_async_returns_same_result() -> None:
    analyzer    = PromptCueAnalyzer()
    sync_result  = analyzer.analyze('compare aurora and opensearch for rag')
    async_result = await analyzer.analyze_async('compare aurora and opensearch for rag')
    assert async_result.primary_query_type == sync_result.primary_query_type
    assert async_result.input_text         == sync_result.input_text


@pytest.mark.asyncio
async def test_warm_up_async_completes() -> None:
    analyzer = PromptCueAnalyzer()
    await analyzer.warm_up_async()
    # After async warm-up, a subsequent analyze call must complete without loading
    result = await analyzer.analyze_async('what is machine learning')
    assert result.primary_query_type != ''
