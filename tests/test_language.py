# promptcue | Unit tests for language detection
# Maintainer: Informity

from __future__ import annotations

from unittest.mock import patch

import pytest

from promptcue.constants import PCUE_UNKNOWN
from promptcue.extraction.language import PromptCueLanguageDetector


class TestLanguageDetector:

    def test_disabled_path_returns_unknown(self) -> None:
        detector = PromptCueLanguageDetector(enabled=False)
        assert detector.detect('This is a perfectly valid English sentence.') == PCUE_UNKNOWN

    def test_short_text_returns_unknown(self) -> None:
        # Text below _MIN_TEXT_LENGTH should not be classified.
        detector = PromptCueLanguageDetector(enabled=True)
        assert detector.detect('hi') == PCUE_UNKNOWN

    def test_empty_string_returns_unknown(self) -> None:
        detector = PromptCueLanguageDetector(enabled=True)
        assert detector.detect('') == PCUE_UNKNOWN

    def test_missing_dep_raises_import_error(self) -> None:
        detector = PromptCueLanguageDetector(enabled=True)
        with patch.dict('sys.modules', {'langdetect': None}):
            # Force _loaded to False so _ensure_lib is called again.
            detector._loaded = False
            with pytest.raises(ImportError, match='promptcue\\[detection\\]'):
                detector.detect('This sentence is long enough to trigger detection.')

    def test_detection_exception_returns_unknown(self) -> None:
        detector = PromptCueLanguageDetector(enabled=True)
        with patch('promptcue.extraction.language.PromptCueLanguageDetector._ensure_lib'):
            detector._loaded = True
            with patch('langdetect.detect', side_effect=Exception('LangDetectException')):
                result = detector.detect('This sentence should trigger detection but fail.')
        assert result == PCUE_UNKNOWN

    def test_is_loaded_false_before_warm_up(self) -> None:
        detector = PromptCueLanguageDetector(enabled=True)
        assert not detector.is_loaded

    def test_warm_up_sets_is_loaded(self) -> None:
        detector = PromptCueLanguageDetector(enabled=True)
        detector.warm_up()
        assert detector.is_loaded

    def test_warm_up_disabled_stays_unloaded(self) -> None:
        detector = PromptCueLanguageDetector(enabled=False)
        detector.warm_up()
        assert not detector.is_loaded

    def test_successful_detection_returns_string(self) -> None:
        detector = PromptCueLanguageDetector(enabled=True)
        result   = detector.detect('How do I configure Redis for production environments?')
        assert isinstance(result, str)
        assert result != ''
