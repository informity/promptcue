# promptcue | Decision engine — resolves classification into a structured result
# Maintainer: Informity

from __future__ import annotations

from dataclasses import dataclass, field

from promptcue.config import PromptCueConfig
from promptcue.constants import (
    PCUE_ACTION_CLARIFY,
    PCUE_BASIS_BELOW_THRESHOLD,
    PCUE_BASIS_TRIGGER_MATCH,
    PCUE_HINT_CLARIFICATION,
    PCUE_HINT_CURRENT_INFO,
    PCUE_HINT_REASONING,
    PCUE_HINT_RETRIEVAL,
    PCUE_UNKNOWN,
)
from promptcue.core.classifier import PromptCueClassificationResult, _top_margin
from promptcue.core.registry import PromptCueRegistry
from promptcue.models.enums import PromptCueConfidenceBand, PromptCueScope


@dataclass(slots=True)
class PromptCueDecisionResult:
    """Resolved output of PromptCueDecisionEngine — the structured classification decision."""
    primary_label:        str
    confidence:           float
    confidence_band:      PromptCueConfidenceBand
    ambiguity_score:      float
    classification_basis: str
    scope:                PromptCueScope
    routing_hints:        dict[str, bool]
    action_hints:         dict[str, bool] = field(default_factory=dict)


class PromptCueDecisionEngine:
    """Converts a raw PromptCueClassificationResult into a final PromptCueDecisionResult.

    Applies similarity thresholds, computes the ambiguity score, resolves
    routing and action hints from the registry, and sets clarification flags
    when the top-two candidates are too close.
    """

    def __init__(self, config: PromptCueConfig, registry: PromptCueRegistry) -> None:
        self.config   = config
        self.registry = registry

    def _confidence_band(
        self,
        score: float,
        basis: str,
    ) -> PromptCueConfidenceBand:
        """Map a raw confidence score to a named band.

        Trigger-match results always map to HIGH — the trigger phrase is an
        explicit intent signal that does not require further calibration.
        Semantic and word-overlap results are mapped by threshold:
          score >= confidence_high_threshold   → HIGH
          score >= confidence_medium_threshold → MEDIUM
          otherwise                            → LOW
        """
        if basis == PCUE_BASIS_TRIGGER_MATCH:
            return PromptCueConfidenceBand.HIGH
        if score >= self.config.confidence_high_threshold:
            return PromptCueConfidenceBand.HIGH
        if score >= self.config.confidence_medium_threshold:
            return PromptCueConfidenceBand.MEDIUM
        return PromptCueConfidenceBand.LOW

    def resolve(
        self,
        result:             PromptCueClassificationResult,
        threshold_override: float | None = None,
    ) -> PromptCueDecisionResult:
        """Resolve a classification result into a final decision.

        Returns unknown + clarification hints when:
        - There are no candidates, or
        - The top score is below the similarity threshold.

        Sets needs_clarification + should_clarify when:
        - The top two candidates are within ambiguity_margin of each other
          (the query is genuinely ambiguous even though it scored above threshold).
        """
        if not result.candidates:
            return PromptCueDecisionResult(
                primary_label        = PCUE_UNKNOWN,
                confidence           = 0.0,
                confidence_band      = PromptCueConfidenceBand.LOW,
                ambiguity_score      = 1.0,
                classification_basis = PCUE_BASIS_BELOW_THRESHOLD,
                scope                = PromptCueScope.UNKNOWN,
                routing_hints        = {
                    PCUE_HINT_CLARIFICATION: True,
                    PCUE_HINT_RETRIEVAL:     False,
                    PCUE_HINT_REASONING:     False,
                    PCUE_HINT_CURRENT_INFO:  False,
                },
                action_hints         = {PCUE_ACTION_CLARIFY: True},
            )

        threshold    = (
            threshold_override if threshold_override is not None
            else self.config.similarity_threshold
        )
        top, margin  = _top_margin(result.candidates)
        # top is always non-None here: the no-candidates guard above already returned.
        if top is None:
            raise RuntimeError('_top_margin returned None despite non-empty candidates list.')
        ambiguity    = max(0.0, min(1.0, 1.0 - margin))
        unclear      = top.score < threshold

        if unclear:
            return PromptCueDecisionResult(
                primary_label        = PCUE_UNKNOWN,
                confidence           = top.score,
                confidence_band      = PromptCueConfidenceBand.LOW,
                ambiguity_score      = ambiguity,
                classification_basis = PCUE_BASIS_BELOW_THRESHOLD,
                scope                = PromptCueScope.UNKNOWN,
                routing_hints        = {
                    PCUE_HINT_CLARIFICATION: True,
                    PCUE_HINT_RETRIEVAL:     False,
                    PCUE_HINT_REASONING:     False,
                    PCUE_HINT_CURRENT_INFO:  False,
                },
                action_hints         = {PCUE_ACTION_CLARIFY: True},
            )

        # Pull routing, scope, and action directives from the registry.
        definition   = self.registry.get_by_label(top.label)
        yaml_routing = definition.routing_hints if definition else {}
        yaml_actions = definition.action_hints  if definition else {}
        # Coerce the YAML string to the typed PromptCueScope enum; fall back to UNKNOWN for any
        # value that is not a recognised scope (e.g. a future custom registry entry).
        _scope_raw   = definition.scope if definition else PromptCueScope.UNKNOWN.value
        try:
            scope = PromptCueScope(_scope_raw)
        except ValueError:
            scope = PromptCueScope.UNKNOWN
        # Per-type margin override — set in query_types_en.yaml as ambiguity_margin_override.
        # Use it when defined; fall back to the global config value otherwise.
        eff_margin   = (
            definition.ambiguity_margin_override
            if definition and definition.ambiguity_margin_override is not None
            else self.config.ambiguity_margin
        )

        # Ambiguous even above threshold — the top two candidates are too close.
        # Keep the top label but flag for clarification.
        is_ambiguous = margin < eff_margin

        routing_hints = {
            PCUE_HINT_CLARIFICATION: is_ambiguous,
            PCUE_HINT_RETRIEVAL:     bool(yaml_routing.get(PCUE_HINT_RETRIEVAL,    False)),
            PCUE_HINT_REASONING:     bool(yaml_routing.get(PCUE_HINT_REASONING,    False)),
            PCUE_HINT_CURRENT_INFO:  bool(yaml_routing.get(PCUE_HINT_CURRENT_INFO, False)),
        }

        # Merge action hints from YAML, then override clarify if ambiguous.
        action_hints: dict[str, bool] = {k: bool(v) for k, v in yaml_actions.items()}
        if is_ambiguous:
            action_hints[PCUE_ACTION_CLARIFY] = True

        return PromptCueDecisionResult(
            primary_label        = top.label,
            confidence           = top.score,
            confidence_band      = self._confidence_band(top.score, top.basis),
            ambiguity_score      = ambiguity,
            classification_basis = top.basis,
            scope                = scope,
            routing_hints        = routing_hints,
            action_hints         = action_hints,
        )
