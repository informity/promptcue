# promptcue | Sentence-transformer embedding backend with lazy loading
# Maintainer: Informity

from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from promptcue.exceptions import PromptCueModelLoadError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class PromptCueEmbeddingBackend:
    """Sentence-transformers embedding backend with lazy, thread-safe model loading.

    The model is loaded on first call to encode() — not at construction time.
    This keeps PromptCueAnalyzer() fast to instantiate and keeps sentence-transformers
    truly optional when semantic scoring is disabled.  A threading.Lock guards the
    lazy-load path so concurrent first requests do not race to load the model twice.

    If the model cannot be loaded (missing cache, no network, bad path), a
    PromptCueModelLoadError is raised immediately.  PromptCue does not degrade to
    deterministic-only mode — that path produces ~40–50% accuracy and is not a
    supported production configuration.
    """

    def __init__(
        self,
        model_name: str       = 'all-MiniLM-L6-v2',
        cache_dir:  Path | None = None,
    ) -> None:
        self._model_name  = model_name
        self._cache_dir   = cache_dir
        self._model: SentenceTransformer | None = None
        self._lock        = threading.Lock()

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
        """Pre-load the sentence-transformer model explicitly.

        This method exists for consumers who construct PromptCueEmbeddingBackend
        directly (e.g. tests, custom pipelines).  In the normal PromptCueAnalyzer
        path, warm-up is handled by PromptCueClassifier.warm_up() which triggers
        model loading via _build_example_cache() → encode() → _ensure_model().
        Raises PromptCueModelLoadError if the model cannot be loaded.
        """
        self._ensure_model()

    # ==============================================================================
    # Internal
    # ==============================================================================

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    'Semantic scoring requires the sentence-transformers package. '
                    'Install it with: pip install "promptcue[semantic]"'
                ) from exc
            cache_path = str(self._cache_dir) if self._cache_dir else '<huggingface default>'
            try:
                kwargs: dict[str, str] = {}
                if self._cache_dir is not None:
                    kwargs['cache_folder'] = str(self._cache_dir)
                self._model = SentenceTransformer(self._model_name, **kwargs)
            except Exception as exc:
                raise PromptCueModelLoadError(self._model_name, cache_path, exc) from exc


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors, clamped to [0.0, 1.0]."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return 0.0
    return float(np.clip(np.dot(a_arr, b_arr) / denom, 0.0, 1.0))


def cosine_similarity_batch(query: list[float], matrix: list[list[float]]) -> list[float]:
    """Vectorised cosine similarity between *query* and every row of *matrix*.

    All dot products are computed in a single NumPy matrix multiply, which is
    significantly faster than calling cosine_similarity() in a Python loop when
    the matrix has many rows (one per example sentence per query type).

    Returns an empty list when *matrix* is empty.
    """
    if not matrix:
        return []
    q = np.array(query,  dtype=np.float32)
    m = np.array(matrix, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    m_norms = np.linalg.norm(m, axis=1)
    denom = q_norm * m_norms
    denom[denom == 0.0] = 1.0   # avoid division by zero; result will be 0 anyway
    return np.clip((m @ q) / denom, 0.0, 1.0).tolist()
