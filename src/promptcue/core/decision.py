# promptcue | Decision engine — resolves classification into a structured result
# Maintainer: Informity

from __future__ import annotations

from dataclasses import dataclass, field

from promptcue.config import PromptCueConfig
from promptcue.constants import (
    PCUE_UNKNOWN,
)
from promptcue.core.classifier import PromptCueClassificationResult, _top_margin
from promptcue.core.registry import PromptCueRegistry
from promptcue.models.enums import (
    PromptCueActionHint,
    PromptCueBasis,
    PromptCueConfidenceBand,
    PromptCueRoutingHint,
    PromptCueScope,
)


@dataclass(slots=True)
class PromptCueDecisionResult:
    """Resolved output of PromptCueDecisionEngine — the structured classification decision."""
    primary_label:        str
    confidence:           float
    confidence_band:      PromptCueConfidenceBand
    ambiguity_score:      float
    type_confidence_margin: float
    scope_confidence:     float
    scope_confidence_margin: float
    classification_basis: PromptCueBasis
    scope:                PromptCueScope
    routing_hints:        dict[PromptCueRoutingHint, bool]
    action_hints:         dict[PromptCueActionHint, bool] = field(default_factory=dict)
    decision_notes:       list[str] = field(default_factory=list)


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
        basis: PromptCueBasis,
    ) -> PromptCueConfidenceBand:
        """Map a raw confidence score to a named band.

        Trigger-match results always map to HIGH — the trigger phrase is an
        explicit intent signal that does not require further calibration.
        Semantic and word-overlap results are mapped by threshold:
          score >= confidence_high_threshold   → HIGH
          score >= confidence_medium_threshold → MEDIUM
          otherwise                            → LOW
        """
        if basis == PromptCueBasis.TRIGGER_MATCH:
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
                type_confidence_margin = 0.0,
                scope_confidence     = 0.0,
                scope_confidence_margin = 0.0,
                classification_basis = PromptCueBasis.BELOW_THRESHOLD,
                scope                = PromptCueScope.UNKNOWN,
                routing_hints        = {
                    PromptCueRoutingHint.NEEDS_CLARIFICATION: True,
                    PromptCueRoutingHint.NEEDS_RETRIEVAL:     False,
                    PromptCueRoutingHint.NEEDS_REASONING:     False,
                    PromptCueRoutingHint.NEEDS_CURRENT_INFO:  False,
                    PromptCueRoutingHint.NEEDS_CHAT_HISTORY:  False,
                },
                action_hints         = {PromptCueActionHint.CLARIFY: True},
                decision_notes       = ['no_candidates'],
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
                type_confidence_margin = margin,
                scope_confidence     = 0.0,
                scope_confidence_margin = margin,
                classification_basis = PromptCueBasis.BELOW_THRESHOLD,
                scope                = PromptCueScope.UNKNOWN,
                routing_hints        = {
                    PromptCueRoutingHint.NEEDS_CLARIFICATION: True,
                    PromptCueRoutingHint.NEEDS_RETRIEVAL:     False,
                    PromptCueRoutingHint.NEEDS_REASONING:     False,
                    PromptCueRoutingHint.NEEDS_CURRENT_INFO:  False,
                    PromptCueRoutingHint.NEEDS_CHAT_HISTORY:  False,
                },
                action_hints         = {PromptCueActionHint.CLARIFY: True},
                decision_notes       = ['below_threshold'],
            )

        # Pull routing, scope, and action directives from the registry.
        definition   = self.registry.get_by_label(top.label)
        yaml_routing = definition.routing_hints if definition else {}
        yaml_actions = definition.action_hints  if definition else {}
        # Coerce the YAML string to the typed PromptCueScope enum; fall back to UNKNOWN for any
        # value that is not a recognised scope (e.g. a future custom registry entry).
        if definition:
            try:
                scope = PromptCueScope(definition.scope)
            except ValueError:
                scope = PromptCueScope.UNKNOWN
        else:
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
            PromptCueRoutingHint.NEEDS_CLARIFICATION: is_ambiguous,
            PromptCueRoutingHint.NEEDS_RETRIEVAL: bool(
                yaml_routing.get(PromptCueRoutingHint.NEEDS_RETRIEVAL.value, False)
            ),
            PromptCueRoutingHint.NEEDS_REASONING: bool(
                yaml_routing.get(PromptCueRoutingHint.NEEDS_REASONING.value, False)
            ),
            PromptCueRoutingHint.NEEDS_CURRENT_INFO: bool(
                yaml_routing.get(PromptCueRoutingHint.NEEDS_CURRENT_INFO.value, False)
            ),
            PromptCueRoutingHint.NEEDS_CHAT_HISTORY: bool(
                yaml_routing.get(PromptCueRoutingHint.NEEDS_CHAT_HISTORY.value, False)
            ),
        }

        # Merge action hints from YAML, then override clarify if ambiguous.
        action_hints: dict[PromptCueActionHint, bool] = {}
        for key, value in yaml_actions.items():
            try:
                action_hints[PromptCueActionHint(key)] = bool(value)
            except ValueError:
                continue
        if is_ambiguous:
            action_hints[PromptCueActionHint.CLARIFY] = True

        decision_notes: list[str] = ['resolved_primary_label']
        if is_ambiguous:
            decision_notes.append('ambiguous_margin')

        return PromptCueDecisionResult(
            primary_label        = top.label,
            confidence           = top.score,
            confidence_band      = self._confidence_band(top.score, top.basis),
            ambiguity_score      = ambiguity,
            type_confidence_margin = margin,
            scope_confidence     = (top.score if scope != PromptCueScope.UNKNOWN else 0.0),
            scope_confidence_margin = margin,
            classification_basis = PromptCueBasis(top.basis),
            scope                = scope,
            routing_hints        = routing_hints,
            action_hints         = action_hints,
            decision_notes       = decision_notes,
        )
