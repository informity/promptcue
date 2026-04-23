# promptcue | Smoke tests for public package imports and enum values
# Maintainer: Informity

from promptcue import (
    PromptCueActionHint,
    PromptCueAnalyzer,
    PromptCueBasis,
    PromptCueConfidenceBand,
    PromptCueConfig,
    PromptCueContinuationSignal,
    PromptCueDiscourseSignal,
    PromptCueEmbedFn,
    PromptCueError,
    PromptCueFollowupSignal,
    PromptCueModelLoadError,
    PromptCueOutputFormat,
    PromptCueQueryObject,
    PromptCueRegistryError,
    PromptCueRoutingHint,
    PromptCueScope,
    PromptCueTopicShiftSignal,
)


def test_public_imports() -> None:
    assert PromptCueAnalyzer is not None
    assert PromptCueConfig is not None
    assert PromptCueQueryObject is not None
    assert PromptCueEmbedFn is not None


def test_exception_imports() -> None:
    assert issubclass(PromptCueModelLoadError, PromptCueError)
    assert issubclass(PromptCueRegistryError, PromptCueError)


def test_enum_imports() -> None:
    assert PromptCueScope.BROAD == "broad"
    assert PromptCueRoutingHint.NEEDS_RETRIEVAL == "needs_retrieval"
    assert PromptCueRoutingHint.NEEDS_STRUCTURE == "needs_structure"
    assert PromptCueActionHint.CONVERSATIONAL == "should_respond_conversationally"
    assert PromptCueBasis.TRIGGER_MATCH == "trigger_match"
    assert PromptCueBasis.BELOW_THRESHOLD == "below_threshold"
    assert PromptCueConfidenceBand.HIGH == "high"
    assert PromptCueOutputFormat.JSON == "json"
    assert PromptCueOutputFormat.TABLE == "table"
    assert PromptCueOutputFormat.YAML == "yaml"
    assert PromptCueDiscourseSignal.PREFIX == "prefix"
    assert PromptCueTopicShiftSignal.EXPLICIT_CUE == "explicit_cue"
    assert PromptCueFollowupSignal.REFERENTIAL == "referential"
    assert PromptCueContinuationSignal.REQUEST == "request"
