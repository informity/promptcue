# promptcue | Enums for scope, routing hints, and action hints — public convenience API
# Maintainer: Informity

from __future__ import annotations

from enum import StrEnum


class PromptCueScope(StrEnum):
    BROAD        = 'broad'
    FOCUSED      = 'focused'
    COMPARATIVE  = 'comparative'
    EXPLORATORY  = 'exploratory'
    UNKNOWN      = 'unknown'


class PromptCueRoutingHint(StrEnum):
    NEEDS_RETRIEVAL     = 'needs_retrieval'
    NEEDS_REASONING     = 'needs_reasoning'
    NEEDS_CURRENT_INFO  = 'needs_current_info'
    NEEDS_CLARIFICATION = 'needs_clarification'


class PromptCueActionHint(StrEnum):
    SURVEY         = 'should_survey'
    ENUMERATE      = 'should_enumerate'
    COMPARE        = 'should_compare'
    DIRECT_ANSWER  = 'should_direct_answer'
    CHECK_RECENCY  = 'should_check_recency'
    CLARIFY        = 'should_clarify'
    CONVERSATIONAL = 'should_respond_conversationally'
