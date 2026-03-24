# promptcue | Language detection using langdetect
# Maintainer: Informity

from __future__ import annotations

from promptcue.constants import PCUE_UNKNOWN


class PromptCueLanguageDetector:
    """Detects the BCP-47 language code of an input string using langdetect.

    The langdetect library is loaded lazily on first detect() call.
    Requires: pip install "promptcue[detection]"

    Returns PCUE_UNKNOWN ('unknown') when:
    - Detection is disabled via enabled=False.
    - The text is too short or ambiguous to classify reliably.
    - The optional dep is not installed.
    """

    # Minimum character count considered reliable for language detection.
    _MIN_TEXT_LENGTH: int = 10

    def __init__(self, enabled: bool = False) -> None:
        self.enabled   = enabled
        self._loaded   = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect(self, text: str) -> str:
        """Return a BCP-47 language code (e.g. 'en', 'fr', 'de') or 'unknown'."""
        if not self.enabled or len(text.strip()) < self._MIN_TEXT_LENGTH:
            return PCUE_UNKNOWN

        self._ensure_lib()
        try:
            from langdetect import detect as _detect
            result = _detect(text)
            return result if isinstance(result, str) else PCUE_UNKNOWN
        except Exception:
            # LangDetectException or any unexpected error → degrade gracefully.
            return PCUE_UNKNOWN

    # ==============================================================================
    # Private
    # ==============================================================================

    def _ensure_lib(self) -> None:
        """Verify langdetect is importable. Raises a clear error if missing."""
        if self._loaded:
            return
        try:
            import langdetect  # noqa: F401
            self._loaded = True
        except ImportError as exc:
            raise ImportError(
                'Language detection requires langdetect. '
                'Install it with: pip install "promptcue[detection]"'
            ) from exc
