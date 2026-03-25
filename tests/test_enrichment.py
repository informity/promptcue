# promptcue | Tests for optional linguistic and keyword enrichment (spaCy, KeyBERT)
# Maintainer: Informity
#
# Each block is skipped automatically when the required optional dep is missing.

import pytest

from promptcue import PromptCueAnalyzer, PromptCueConfig
from promptcue.extraction.keywords import PromptCueKeywordExtractor
from promptcue.extraction.linguistic import PromptCueLinguisticExtractor

_QUERY = 'Compare Aurora and OpenSearch for RAG workloads on AWS'


# ==============================================================================
# Graceful failure when optional dep is missing
# ==============================================================================

def test_linguistic_disabled_returns_empty() -> None:
    extractor = PromptCueLinguisticExtractor(enabled=False)
    result    = extractor.extract(_QUERY)
    assert result.main_verbs     == []
    assert result.noun_phrases   == []
    assert result.named_entities == []


def test_keyword_disabled_returns_empty() -> None:
    extractor = PromptCueKeywordExtractor(enabled=False)
    assert extractor.extract(_QUERY) == []


def test_linguistic_missing_dep_raises_import_error() -> None:
    import sys

    # Temporarily hide spaCy to simulate a missing optional dep.
    original = sys.modules.get('spacy')
    sys.modules['spacy'] = None  # type: ignore[assignment]
    try:
        extractor       = PromptCueLinguisticExtractor(enabled=True)
        extractor._nlp  = None  # reset any cached model
        with pytest.raises(ImportError, match='promptcue\\[linguistic\\]'):
            extractor.extract(_QUERY)
    finally:
        if original is None:
            del sys.modules['spacy']
        else:
            sys.modules['spacy'] = original


def test_keyword_missing_dep_raises_import_error() -> None:
    import sys

    original = sys.modules.get('keybert')
    sys.modules['keybert'] = None  # type: ignore[assignment]
    try:
        extractor          = PromptCueKeywordExtractor(enabled=True)
        extractor._kw_model = None
        with pytest.raises(ImportError, match='promptcue\\[keywords\\]'):
            extractor.extract(_QUERY)
    finally:
        if original is None:
            del sys.modules['keybert']
        else:
            sys.modules['keybert'] = original


# ==============================================================================
# spaCy linguistic extraction (skipped if spaCy not installed)
# ==============================================================================

def _spacy_model_available() -> bool:
    try:
        import spacy
        spacy.load('en_core_web_sm')
        return True
    except (ImportError, OSError):
        return False

if not _spacy_model_available():
    pytest.skip(
        'spaCy or en_core_web_sm model not available — skipping linguistic tests',
        allow_module_level=True,
    )


@pytest.fixture(scope='module')
def linguistic_analyzer() -> PromptCueAnalyzer:
    analyzer = PromptCueAnalyzer(PromptCueConfig(enable_linguistic_extraction=True))
    analyzer.warm_up()
    return analyzer


def test_linguistic_extracts_noun_phrases(linguistic_analyzer: PromptCueAnalyzer) -> None:
    result = linguistic_analyzer.analyze(_QUERY)
    assert len(result.noun_phrases) > 0, 'Expected at least one noun phrase'


def test_linguistic_extracts_named_entities(linguistic_analyzer: PromptCueAnalyzer) -> None:
    # Aurora, OpenSearch, and AWS are recognisable entities in en_core_web_sm.
    result = linguistic_analyzer.analyze(_QUERY)
    assert len(result.named_entities) > 0, 'Expected at least one named entity'


def test_linguistic_extracts_main_verbs(linguistic_analyzer: PromptCueAnalyzer) -> None:
    # Use a clear imperative/action query so spaCy tags the root as VERB not AUX.
    result = linguistic_analyzer.analyze('How do I configure CloudWatch log retention policies?')
    # "configure" is the root VERB in this sentence.
    assert len(result.main_verbs) > 0, 'Expected at least one main verb'


def test_linguistic_is_loaded_after_warm_up(linguistic_analyzer: PromptCueAnalyzer) -> None:
    assert linguistic_analyzer.linguistic_extractor.is_loaded


def test_linguistic_empty_text_returns_empty() -> None:
    extractor = PromptCueLinguisticExtractor(enabled=True)
    result    = extractor.extract('   ')
    assert result.main_verbs     == []
    assert result.noun_phrases   == []
    assert result.named_entities == []


def test_linguistic_respects_custom_spacy_model() -> None:
    extractor = PromptCueLinguisticExtractor(enabled=True, model_name='en_core_web_sm')
    result    = extractor.extract(_QUERY)
    assert isinstance(result.noun_phrases, list)


# ==============================================================================
# KeyBERT keyword extraction (skipped if keybert not installed)
# ==============================================================================

keybert_mod = pytest.importorskip(
    'keybert', reason='KeyBERT not installed — skipping keyword tests',
)


@pytest.fixture(scope='module')
def keyword_analyzer() -> PromptCueAnalyzer:
    analyzer = PromptCueAnalyzer(PromptCueConfig(enable_keyword_extraction=True, max_keywords=5))
    analyzer.warm_up()
    return analyzer


def test_keyword_extraction_returns_keywords(keyword_analyzer: PromptCueAnalyzer) -> None:
    result = keyword_analyzer.analyze(_QUERY)
    assert len(result.keywords) > 0, 'Expected at least one keyword'


def test_keyword_weight_in_range(keyword_analyzer: PromptCueAnalyzer) -> None:
    result = keyword_analyzer.analyze(_QUERY)
    for kw in result.keywords:
        assert 0.0 <= kw.weight <= 1.0, f'Weight out of range: {kw.weight}'


def test_keyword_kind_is_keyphrase(keyword_analyzer: PromptCueAnalyzer) -> None:
    result = keyword_analyzer.analyze(_QUERY)
    assert all(kw.kind == 'keyphrase' for kw in result.keywords)


def test_keyword_count_respects_max(keyword_analyzer: PromptCueAnalyzer) -> None:
    result = keyword_analyzer.analyze(_QUERY)
    assert len(result.keywords) <= 5


def test_keyword_is_loaded_after_warm_up(keyword_analyzer: PromptCueAnalyzer) -> None:
    assert keyword_analyzer.keyword_extractor.is_loaded


def test_keyword_empty_text_returns_empty() -> None:
    extractor = PromptCueKeywordExtractor(enabled=True)
    assert extractor.extract('   ') == []
