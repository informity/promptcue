# promptcue | Smoke tests for public package imports and enum values
# Maintainer: Informity

from promptcue import (
    PromptCueActionHint,
    PromptCueAnalyzer,
    PromptCueBasis,
    PromptCueConfig,
    PromptCueQueryObject,
    PromptCueRoutingHint,
    PromptCueScope,
)


def test_public_imports() -> None:
    assert PromptCueAnalyzer    is not None
    assert PromptCueConfig      is not None
    assert PromptCueQueryObject is not None


def test_enum_imports() -> None:
    assert PromptCueScope.BROAD                        == 'broad'
    assert PromptCueRoutingHint.NEEDS_RETRIEVAL        == 'needs_retrieval'
    assert PromptCueActionHint.CONVERSATIONAL          == 'should_respond_conversationally'
    assert PromptCueBasis.TRIGGER_MATCH                == 'trigger_match'
    assert PromptCueBasis.BELOW_THRESHOLD              == 'below_threshold'
