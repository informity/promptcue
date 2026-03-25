# promptcue | End-to-end smoke test — verifies the default analyzer returns a valid result
# Maintainer: Informity

from promptcue import PromptCueAnalyzer


def test_analyzer_smoke() -> None:
    analyzer = PromptCueAnalyzer()
    result   = analyzer.analyze('Compare Aurora and OpenSearch for RAG')
    assert result.query_type == 'comparison'
    assert result.confidence > 0
