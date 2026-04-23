# promptcue | Enums for scope, routing hints, and action hints — public convenience API
# Maintainer: Informity

from __future__ import annotations

from enum import StrEnum


class PromptCueScope(StrEnum):
    BROAD = "broad"
    COMPARATIVE = "comparative"
    EXPLORATORY = "exploratory"
    FOCUSED = "focused"
    UNKNOWN = "unknown"


class PromptCueConfidenceBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PromptCueRoutingHint(StrEnum):
    NEEDS_CHAT_HISTORY = "needs_chat_history"
    NEEDS_CLARIFICATION = "needs_clarification"
    NEEDS_CURRENT_INFO = "needs_current_info"
    NEEDS_REASONING = "needs_reasoning"
    NEEDS_RETRIEVAL = "needs_retrieval"
    NEEDS_STRUCTURE = "needs_structure"


class PromptCueActionHint(StrEnum):
    CHECK_RECENCY = "should_check_recency"
    CLARIFY = "should_clarify"
    COMPARE = "should_compare"
    DIRECT_ANSWER = "should_direct_answer"
    ENUMERATE = "should_enumerate"
    CONVERSATIONAL = "should_respond_conversationally"
    SURVEY = "should_survey"


class PromptCueOutputFormat(StrEnum):
    BULLETS = "bullets"
    CSV = "csv"
    JSON = "json"
    LIST = "list"
    NARRATIVE = "narrative"
    TABLE = "table"
    YAML = "yaml"


class PromptCueDiscourseSignal(StrEnum):
    NONE = "none"
    PREFIX = "prefix"


class PromptCueTopicShiftSignal(StrEnum):
    NONE = "none"
    EXPLICIT_CUE = "explicit_cue"


class PromptCueFollowupSignal(StrEnum):
    NONE = "none"
    REFERENTIAL = "referential"


class PromptCueContinuationSignal(StrEnum):
    NONE = "none"
    REQUEST = "request"


class PromptCueBasis(StrEnum):
    """Typed equivalent of the PCUE_BASIS_* constants in constants.py."""

    BELOW_THRESHOLD = "below_threshold"
    FALLBACK = "fallback"
    SEMANTIC = "semantic_similarity"
    TRIGGER_MATCH = "trigger_match"
    WORD_OVERLAP = "word_overlap"
