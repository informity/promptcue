# promptcue | PromptCue package-level constants
# Maintainer: Informity

from __future__ import annotations

from pathlib import Path

# ==============================================================================
# Package directory layout
# ==============================================================================

PCUE_PACKAGE_DIR      = Path(__file__).parent
PCUE_DATA_DIR         = PCUE_PACKAGE_DIR / 'data'
PCUE_DEFAULT_REGISTRY = PCUE_DATA_DIR / 'query_types_en.yaml'

# ==============================================================================
# Schema versioning
# ==============================================================================

PCUE_SCHEMA_VERSION   = '1.0'

# ==============================================================================
# Sentinel / placeholder values
# ==============================================================================

# PCUE_UNKNOWN is the sentinel for primary_query_type when the query could not be
# classified above the confidence threshold.  Do not use it for scope — use
# PromptCueScope.UNKNOWN (or PCUE_SCOPE_UNKNOWN below) for that field instead,
# so that the two can evolve independently if scope values ever diverge.
PCUE_UNKNOWN          = 'unknown'

# ==============================================================================
# Classification basis strings — explain how the top candidate was matched
# Typed equivalent: PromptCueBasis (StrEnum) in models/enums.py
# ==============================================================================

PCUE_BASIS_TRIGGER_MATCH   = 'trigger_match'       # trigger phrase matched; score ∝ length
PCUE_BASIS_WORD_OVERLAP    = 'word_overlap'        # vocabulary overlap with type definition
PCUE_BASIS_SEMANTIC        = 'semantic_similarity' # embedding cosine similarity
PCUE_BASIS_FALLBACK        = 'fallback'            # no vocabulary overlap; floor score (0.10)
PCUE_BASIS_BELOW_THRESHOLD = 'below_threshold'     # top score < similarity_threshold → unknown

# ==============================================================================
# Known routing hint keys — must mirror keys used in query_types_en.yaml
# ==============================================================================

PCUE_HINT_RETRIEVAL     = 'needs_retrieval'
PCUE_HINT_REASONING     = 'needs_reasoning'
PCUE_HINT_CURRENT_INFO  = 'needs_current_info'
PCUE_HINT_CLARIFICATION = 'needs_clarification'
PCUE_HINT_STRUCTURE     = 'needs_structure'

# ==============================================================================
# Scope values — must mirror values used in query_types_en.yaml and PromptCueScope enum
# ==============================================================================

PCUE_SCOPE_BROAD       = 'broad'
PCUE_SCOPE_FOCUSED     = 'focused'
PCUE_SCOPE_COMPARATIVE = 'comparative'
PCUE_SCOPE_EXPLORATORY = 'exploratory'
# PCUE_SCOPE_UNKNOWN is the sentinel for the scope field specifically.  Its value
# happens to be identical to PCUE_UNKNOWN today, but they are kept as separate
# constants so a change to one does not silently affect the other.  Prefer
# PromptCueScope.UNKNOWN in new internal code.
PCUE_SCOPE_UNKNOWN     = 'unknown'

# ==============================================================================
# Known action hint keys — must mirror keys used in query_types_en.yaml
# These guide response generation: how the LLM should structure its answer.
# ==============================================================================

PCUE_ACTION_SURVEY         = 'should_survey'          # broad overview / landscape response
PCUE_ACTION_ENUMERATE      = 'should_enumerate'        # numbered steps or bullet list
PCUE_ACTION_COMPARE        = 'should_compare'          # side-by-side or structured tradeoff
PCUE_ACTION_DIRECT         = 'should_direct_answer'    # single concise factual answer
PCUE_ACTION_CHECK_RECENCY  = 'should_check_recency'    # warn about data freshness
PCUE_ACTION_CLARIFY        = 'should_clarify'          # ask for more context before answering
PCUE_ACTION_CONVERSATIONAL = 'should_respond_conversationally'  # casual / social tone
