# promptcue | Enums for scope and routing hints — public convenience API
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
