# promptcue | Cascade classifier — deterministic + semantic scoring
# Maintainer: Informity

from __future__ import annotations

from dataclasses import dataclass

from promptcue.config import PromptCueConfig
from promptcue.constants import (
    PCUE_BASIS_FALLBACK,
    PCUE_BASIS_LABEL_MATCH,
    PCUE_BASIS_SEMANTIC,
    PCUE_BASIS_TRIGGER_MATCH,
    PCUE_BASIS_WORD_OVERLAP,
)
from promptcue.core.embedding import PromptCueEmbeddingBackend, cosine_similarity
from promptcue.core.registry import PromptCueRegistry
from promptcue.models.schema import PromptCueCandidate


@dataclass(slots=True)
class PromptCueClassificationResult:
    candidates: list[PromptCueCandidate]


class PromptCueClassifier:
    """Scores a query against all registered query types.

    Implements a cascade strategy: the deterministic pass always runs first.
    The semantic pass (embedding cosine similarity) fires as a fallback only
    when the deterministic result is below semantic_fallback_threshold or the
    top-two candidates are within ambiguity_margin.
    """

    def __init__(self, registry: PromptCueRegistry, config: PromptCueConfig) -> None:
        self.registry          = registry
        self.config            = config
        self.embedding_backend = PromptCueEmbeddingBackend(model_name=config.embedding_model)
        # Cached per-label example embeddings; populated lazily on first semantic classify.
        self._example_cache: dict[str, list[list[float]]] = {}

    # ==============================================================================
    # Public
    # ==============================================================================

    def classify(self, text: str) -> PromptCueClassificationResult:
        """Cascade classifier — deterministic first, semantic fallback.

        The deterministic pass runs unconditionally and is returned immediately
        when EITHER of the following holds:
          • semantic scoring is disabled (pure-deterministic mode), OR
          • the top deterministic score reaches semantic_fallback_threshold AND
            the gap between top-1 and top-2 meets the ambiguity_margin.

        Every other case falls through to the semantic pass, which re-scores
        using sentence-level embeddings and supersedes the deterministic result.
        """
        det = self._classify_deterministic(text)

        if not self.config.enable_semantic_scoring:
            return det

        top          = det.candidates[0] if det.candidates else None
        second_score = det.candidates[1].score if len(det.candidates) > 1 else 0.0
        margin       = (top.score - second_score) if top else 0.0

        high_confidence = (
            top is not None
            and top.score >= self.config.semantic_fallback_threshold
            and margin   >= self.config.ambiguity_margin
        )
        if high_confidence:
            return det

        return self._classify_semantic(text)

    def warm_up(self) -> None:
        """Pre-load the embedding model and pre-compute all example embeddings.

        No-op when semantic scoring is disabled — safe to call in all installs.
        """
        if self.config.enable_semantic_scoring:
            self._build_example_cache()

    # ==============================================================================
    # Deterministic path
    # ==============================================================================

    def _classify_deterministic(self, text: str) -> PromptCueClassificationResult:
        """Score every registered type against *text* using three tiers:

        Tier 1 — label_match (0.90):
            The type label itself appears verbatim in the query.

        Tier 2 — trigger_match (0.60–0.85):
            One or more trigger phrases are substrings of the query.
            Score is proportional to the length of the longest matching trigger:
            short/vague triggers ("how do I") score lower than long/specific ones
            ("how do I configure and deploy"), so overlapping triggers resolve
            to the most specific match rather than a coin-flip tie.

        Tier 3 — word_overlap (0.10–0.50):
            No trigger matched, but query words appear in the type's vocabulary
            (triggers + examples + description).  Provides a graded soft signal
            instead of a flat fallback, making candidate_query_types useful for
            downstream consumers even when no type wins outright.

        Floor — fallback (0.10):
            Zero vocabulary overlap.  Score is the same floor as before but the
            basis is explicit.
        """
        scores:  list[PromptCueCandidate] = []
        lowered: str                 = text.lower()

        # Pre-compute query content words (length > 2 filters most stop-words).
        query_words: frozenset[str] = frozenset(w for w in lowered.split() if len(w) > 2)

        for definition in self.registry.definitions:
            match_phrases = definition.triggers if definition.triggers else definition.examples

            if definition.label.lower() in lowered:
                score = 0.90
                basis = PCUE_BASIS_LABEL_MATCH

            else:
                matched = [p for p in match_phrases if p.lower() in lowered]
                if matched:
                    # Longest match wins — proxy for trigger specificity.
                    best        = max(matched, key=len)
                    specificity = min(len(best) / 35.0, 1.0)   # normalise; 35 chars ≈ long trigger
                    score = round(0.60 + 0.25 * specificity, 4)
                    basis = PCUE_BASIS_TRIGGER_MATCH
                else:
                    # Soft word-overlap fallback across all type vocabulary.
                    type_text  = ' '.join(
                        definition.triggers + definition.examples + [definition.description]
                    )
                    type_words = frozenset(w for w in type_text.lower().split() if len(w) > 2)
                    overlap    = (
                        len(query_words & type_words) / len(query_words)
                        if query_words else 0.0
                    )
                    if overlap > 0.0:
                        score = round(min(0.10 + 0.40 * overlap, 0.50), 4)
                        basis = PCUE_BASIS_WORD_OVERLAP
                    else:
                        score = 0.10
                        basis = PCUE_BASIS_FALLBACK

            scores.append(PromptCueCandidate(
                label=definition.label, score=min(score, 1.0), basis=basis,
            ))

        scores.sort(key=lambda item: item.score, reverse=True)
        return PromptCueClassificationResult(candidates=scores)

    # ==============================================================================
    # Semantic path (sentence-transformers)
    # ==============================================================================

    def _classify_semantic(self, text: str) -> PromptCueClassificationResult:
        """Score every registered type against *text* using embedding cosine similarity.

        Each type's examples are encoded once and cached.  The best (max) cosine
        similarity across a type's example embeddings becomes that type's score.
        """
        query_vec = self.embedding_backend.encode([text])[0]
        cache     = self._build_example_cache()
        scores: list[PromptCueCandidate] = []

        for definition in self.registry.definitions:
            example_vecs = cache.get(definition.label, [])
            if not example_vecs:
                scores.append(PromptCueCandidate(
                    label=definition.label, score=0.0, basis=PCUE_BASIS_FALLBACK,
                ))
                continue

            best_sim = max(cosine_similarity(query_vec, ex_vec) for ex_vec in example_vecs)
            scores.append(PromptCueCandidate(
                label=definition.label,
                score=round(min(max(best_sim, 0.0), 1.0), 6),
                basis=PCUE_BASIS_SEMANTIC,
            ))

        scores.sort(key=lambda item: item.score, reverse=True)
        return PromptCueClassificationResult(candidates=scores)

    def _build_example_cache(self) -> dict[str, list[list[float]]]:
        """Build and return the per-label example embedding cache.

        Encodes all examples from the registry on first call; subsequent calls
        return the already-populated cache.  Requires sentence-transformers.
        """
        if self._example_cache:
            return self._example_cache

        for definition in self.registry.definitions:
            if definition.examples:
                self._example_cache[definition.label] = self.embedding_backend.encode(
                    definition.examples
                )
        return self._example_cache
