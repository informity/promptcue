# promptcue | Sentence-transformer embedding backend with lazy loading
# Maintainer: Informity

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class PromptCueEmbeddingBackend:
    """Sentence-transformers embedding backend with lazy model loading.

    The model is loaded on first call to encode() — not at construction time.
    This keeps PromptCueAnalyzer() fast to instantiate and keeps sentence-transformers
    truly optional when semantic scoring is disabled.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2') -> None:
        self._model_name  = model_name
        self._model: SentenceTransformer | None = None

    # ==============================================================================
    # Public interface
    # ==============================================================================

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        """Encode texts and return a list of float vectors."""
        self._ensure_model()
        embeddings = self._model.encode(list(texts), convert_to_numpy=True)  # type: ignore[union-attr]
        return embeddings.tolist()

    def warm_up(self) -> None:
        """Pre-load the model explicitly. Call this to pay the load cost upfront."""
        self._ensure_model()

    # ==============================================================================
    # Internal
    # ==============================================================================

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                'Semantic scoring requires the sentence-transformers package. '
                'Install it with: pip install "promptcue[semantic]"'
            ) from exc
        self._model = SentenceTransformer(self._model_name)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors, clamped to [0.0, 1.0]."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return 0.0
    return float(np.clip(np.dot(a_arr, b_arr) / denom, 0.0, 1.0))
