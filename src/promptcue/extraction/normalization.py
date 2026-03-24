# promptcue | Text normalisation utilities
# Maintainer: Informity

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r'\s+')


def normalize_text(text: str) -> str:
    """Normalise Unicode and collapse whitespace in *text*."""
    normalized = unicodedata.normalize('NFKC', text)
    normalized = _WHITESPACE_RE.sub(' ', normalized).strip()
    return normalized
