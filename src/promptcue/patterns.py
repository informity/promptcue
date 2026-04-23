# promptcue | Generic prompt pattern constants
# Maintainer: Informity

from __future__ import annotations

import re

YEAR_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"\b(?:19|20)\d{2}\b")

# Openers that indicate the query is a follow-up to a previous turn.
CONTINUATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*also[,\s]", re.IGNORECASE),
    re.compile(r"^\s*and\s+(what\s+about|also|another)\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+about\s", re.IGNORECASE),
    re.compile(r"^\s*furthermore[,\s]", re.IGNORECASE),
    re.compile(r"^\s*additionally[,\s]", re.IGNORECASE),
    re.compile(r"^\s*following\s+up\b", re.IGNORECASE),
    re.compile(r"^\s*oh\s+and\b", re.IGNORECASE),
    re.compile(r"^\s*one\s+more\s+(thing|question)\b", re.IGNORECASE),
    re.compile(r"^\s*building\s+on\s+(that|this|what)", re.IGNORECASE),
    re.compile(r"^\s*to\s+follow\s+up\b", re.IGNORECASE),
    re.compile(r"^\s*going\s+back\s+to\b", re.IGNORECASE),
    re.compile(r"^\s*on\s+that\s+(note|topic|subject)\b", re.IGNORECASE),
    re.compile(r"^\s*related\s+to\s+that\b", re.IGNORECASE),
]

# Patterns indicating the caller wants a specific output structure.
STRUCTURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"#{1,6}\s+\w", re.MULTILINE),
    re.compile(r"\bformat\s+(this\s+)?as\b", re.IGNORECASE),
    re.compile(r"\boutput\s+(this\s+)?as\s+(a\s+)?table\b", re.IGNORECASE),
    re.compile(r"\bin\s+a\s+(markdown\s+)?table\b", re.IGNORECASE),
    re.compile(r"\bas\s+a\s+table\b", re.IGNORECASE),
    re.compile(r"\bformatted\s+as\b", re.IGNORECASE),
    re.compile(r"\bin\s+bullet\s+(points?|form)\b", re.IGNORECASE),
    re.compile(r"\bwith\s+(?:the\s+following\s+)?sections?:", re.IGNORECASE),
    re.compile(r"\borganiz(?:e|ed)\s+(?:it\s+)?as\b", re.IGNORECASE),
    re.compile(r"\bstructur(?:e|ed)\s+(?:it\s+)?as\b", re.IGNORECASE),
    re.compile(r"\bpresent\s+(?:this|it)\s+as\b", re.IGNORECASE),
]

MULTI_ITEM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        (
            r"\b(across all|across the entire|all (?:documents|files|records|sources|entries)"
            r"|multiple (?:documents|files|records|sources|entries))\b"
        ),
        re.IGNORECASE,
    ),
    re.compile(r"\b(documents|files|records|sources|entries|items)\b", re.IGNORECASE),
    re.compile(r"\bmost important (?:dates|amounts|figures|names)\b", re.IGNORECASE),
]

SYNTHESIS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(summarize|summary|overview|key findings?|main findings?)\b", re.IGNORECASE),
    re.compile(r"\b(tell me about|describe)\b", re.IGNORECASE),
]

COMPARISON_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(compare|contrast|versus|vs\.?|trade[-\s]*offs?|pros?\s+and\s+cons?)\b", re.IGNORECASE
    ),
]

ENUMERATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(list|enumerate|step[-\s]*by[-\s]*step|top\s+\d+|bullet(?:s| points?)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        (
            r"\bwhat\s+are\s+the\s+"
            r"(?:(?:most|key)\s+important\s+)?"
            r"(?:names?|people|dates?|amounts?|figures?|values?)\b"
        ),
        re.IGNORECASE,
    ),
]

# Generic discourse / context semantics.
DISCOURSE_PREFIX_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?:"
    r"ok(?:ay)?|alright|well|so|anyway|now|"
    r"(?:new|different)\s+(?:topic|subject)s?|"
    r"on\s+another\s+subject|"
    r"switch(?:ing)?\s+(?:topic|topics|subject|subjects|context)|"
    r"(?:back|return(?:ing)?)\s+to"
    r")\b",
    re.IGNORECASE,
)

TOPIC_SHIFT_CUE_PATTERN: re.Pattern[str] = re.compile(
    r"\bnew\s+topic\b"
    r"|"
    r"\bchange\s+(?:the\s+)?(?:topic|subject)\b"
    r"|"
    r"\bswitch(?:ing)?\s+(?:topic|topics|context)\b"
    r"|"
    r"\bdifferent\s+topic\b"
    r"|"
    r"\binstead\b"
    r"|"
    r"\bunrelated\b"
    r"|"
    r"\bnow\s+(?:about|switch(?:ing)?)\b",
    re.IGNORECASE,
)

REFERENTIAL_FOLLOWUP_PATTERN: re.Pattern[str] = re.compile(
    r"\b("
    r"there|that|those|these|it|they|them|same|above|earlier|previous|prior|"
    r"as\s+discussed|as\s+mentioned|continue|follow[-\s]?up|again"
    r")\b",
    re.IGNORECASE,
)

CONTINUATION_REQUEST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(continue|go\s+on|keep\s+going|the\s+rest|tell\s+me\s+more|next\s+part|next\s+section)\b",
        re.IGNORECASE,
    ),
]

_MULTI_PERIOD_REGEX: tuple[str, ...] = (
    r"\byear[-\s]*(?:by|over)[-\s]*year\b",
    r"\bcross[-\s]*year\b",
    r"\bby\s+year\b",
    r"\bover\s+time\b",
    r"\b(?:quarterly|annual|monthly)\s+trend\b",
    r"\b(?:over|in|for)\s+the\s+(?:last|past)\s+\d+\s+years?\b",
    r"\bfrom\s+(?:19|20)\d{2}\s+to\s+(?:19|20)\d{2}\b",
    r"\bbetween\s+(?:19|20)\d{2}\s+and\s+(?:19|20)\d{2}\b",
)

TEMPORAL_SCOPE_PATTERNS: list[re.Pattern[str]] = [
    YEAR_TOKEN_PATTERN,
    *(re.compile(pattern, re.IGNORECASE) for pattern in _MULTI_PERIOD_REGEX),
    re.compile(r"\byears?\s+covered\b", re.IGNORECASE),
    re.compile(r"\byear[-\s]*to[-\s]*date\b", re.IGNORECASE),
    re.compile(r"\bYTD\b"),
    re.compile(r"\bsince\s+(?:19|20)\d{2}\b", re.IGNORECASE),
    re.compile(r"\bstarting\s+(?:from\s+)?(?:19|20)\d{2}\b", re.IGNORECASE),
    re.compile(r"\bQ[1-4]\s+(?:19|20)\d{2}\b"),
    re.compile(r"\b(?:19|20)\d{2}\s+Q[1-4]\b"),
]

EXPLICIT_RECENCY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\btoday\b", re.IGNORECASE),
    re.compile(r"\btomorrow\b", re.IGNORECASE),
    re.compile(r"\byesterday\b", re.IGNORECASE),
    re.compile(r"\bright\s+now\b", re.IGNORECASE),
    re.compile(r"\bnow\b", re.IGNORECASE),
    re.compile(r"\bcurrently\b", re.IGNORECASE),
    re.compile(r"\b(?:who|what|when|where)\s+(?:is|are)\s+(?:the\s+)?current\b", re.IGNORECASE),
    re.compile(
        r"\bcurrent\s+(?:state|status|version|year|quarter|month|week|price|rate|events?|conditions?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\blatest\b", re.IGNORECASE),
    re.compile(r"\bmost\s+recent\b", re.IGNORECASE),
    re.compile(r"\brecent\s+(?:updates?|changes?|events?|developments?|news)\b", re.IGNORECASE),
    re.compile(r"\bup[-\s]*to[-\s]*date\b", re.IGNORECASE),
    re.compile(r"\breal[-\s]*time\b", re.IGNORECASE),
    re.compile(r"\blive\b", re.IGNORECASE),
    re.compile(r"\bthis\s+week\b", re.IGNORECASE),
    re.compile(r"\bthis\s+month\b", re.IGNORECASE),
    re.compile(r"\bthis\s+year\b", re.IGNORECASE),
]

MULTI_PERIOD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE) for pattern in _MULTI_PERIOD_REGEX
]

OUTPUT_FORMAT_PATTERNS: dict[str, re.Pattern[str]] = {
    "bullets": re.compile(
        r"\b(as\s+bullet\s+points?|in\s+bullet\s+points?|bullet\s+list|as\s+bullets?)\b",
        re.IGNORECASE,
    ),
    "csv": re.compile(r"\b(csv\s+format|as\s+csv|output\s+csv)\b", re.IGNORECASE),
    "list": re.compile(r"\b(as\s+a\s+list|list\s+format)\b", re.IGNORECASE),
    "narrative": re.compile(r"\b(in\s+narrative\s+form|as\s+paragraphs?)\b", re.IGNORECASE),
    "json": re.compile(r"\b(as\s+json|json\s+format|output\s+json|in\s+json)\b", re.IGNORECASE),
    "table": re.compile(
        r"\b(markdown\s+table|as\s+a\s+table|in\s+table\s+form|in\s+columns?)\b", re.IGNORECASE
    ),
    "yaml": re.compile(r"\b(as\s+ya?ml|ya?ml\s+format|output\s+ya?ml|in\s+ya?ml)\b", re.IGNORECASE),
}
