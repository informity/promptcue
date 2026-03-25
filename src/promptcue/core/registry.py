# promptcue | Query type registry — loads and validates query_types_en.yaml
# Maintainer: Informity

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from promptcue.constants import PCUE_DEFAULT_REGISTRY, PCUE_SCOPE_UNKNOWN
from promptcue.exceptions import PromptCueRegistryError


@dataclass(slots=True)
class PromptCueTypeDefinition:
    """A single query type entry loaded from the YAML registry."""
    label:                    str
    description:              str
    triggers:                 list[str]       # short phrases for deterministic substring matching
    examples:                 list[str]       # full sentences for semantic embedding anchors
    negatives:                list[str]       # sentences this type should NOT match (penalty)
    routing_hints:            dict[str, bool]
    scope:                    str             # broad | focused | comparative | exploratory
    action_hints:             dict[str, bool] # response-generation directives
    ambiguity_margin_override: float | None = None  # overrides PromptCueConfig.ambiguity_margin


class PromptCueRegistry:
    """Holds the full set of query type definitions loaded from a YAML file.

    Can be constructed from a YAML path (via from_yaml) or from an explicit
    list of PromptCueTypeDefinition objects (for testing).  Validates on construction.
    """

    # Class-level annotations so mypy resolves instance attributes set in from_yaml()
    # via cls.__new__(), which bypasses __init__.
    definitions: list[PromptCueTypeDefinition]
    _by_label:   dict[str, PromptCueTypeDefinition]

    def __init__(self, definitions: list[PromptCueTypeDefinition] | None = None) -> None:
        if definitions is None:
            loaded           = self.from_yaml(PCUE_DEFAULT_REGISTRY)
            self.definitions = loaded.definitions
            self._by_label   = loaded._by_label
        else:
            self._validate(definitions)
            self.definitions = definitions
            self._by_label   = {defn.label: defn for defn in definitions}

    # ==============================================================================
    # Queries
    # ==============================================================================

    def get_query_types(self) -> list[PromptCueTypeDefinition]:
        """Return a copy of all registered type definitions."""
        return list(self.definitions)

    def get_by_label(self, label: str) -> PromptCueTypeDefinition | None:
        """Return the definition for *label*, or None if not found."""
        return self._by_label.get(label)

    # ==============================================================================
    # Loading
    # ==============================================================================

    @classmethod
    def from_yaml(cls, path: Path) -> PromptCueRegistry:
        try:
            raw = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise PromptCueRegistryError(f'Unable to load query type registry from {path}') from exc

        if not isinstance(raw, dict):
            raise PromptCueRegistryError(
                f'Registry file {path} is empty or does not contain a valid YAML mapping.'
            )

        definitions = [
            PromptCueTypeDefinition(
                label                     = item['label'],
                description               = item.get('description', ''),
                triggers                  = item.get('triggers', []),
                examples                  = item.get('examples', []),
                negatives                 = item.get('negatives', []),
                routing_hints             = item.get('routing_hints', {}),
                scope                     = item.get('scope', PCUE_SCOPE_UNKNOWN),
                action_hints              = item.get('action_hints', {}),
                ambiguity_margin_override = item.get('ambiguity_margin_override'),
            )
            for item in raw.get('query_types', [])
        ]
        instance = cls.__new__(cls)
        instance._validate(definitions)
        instance.definitions = definitions
        instance._by_label   = {defn.label: defn for defn in definitions}
        return instance

    # ==============================================================================
    # Validation
    # ==============================================================================

    @staticmethod
    def _validate(definitions: list[PromptCueTypeDefinition]) -> None:
        seen_labels: set[str] = set()
        for defn in definitions:
            if not defn.label or not isinstance(defn.label, str):
                raise PromptCueRegistryError(
                    'Each query type entry must have a non-empty string label.'
                )
            if defn.label in seen_labels:
                raise PromptCueRegistryError(f'Duplicate label in registry: "{defn.label}".')
            seen_labels.add(defn.label)

            if not defn.description or not isinstance(defn.description, str):
                raise PromptCueRegistryError(
                    f'Entry "{defn.label}" must have a non-empty description.'
                )

            if not defn.examples or not isinstance(defn.examples, list):
                raise PromptCueRegistryError(
                    f'Entry "{defn.label}" must have at least one example sentence.'
                )
            for ex in defn.examples:
                if not isinstance(ex, str) or not ex.strip():
                    raise PromptCueRegistryError(
                        f'Entry "{defn.label}" contains an empty or non-string example.'
                    )
            for tr in defn.triggers:
                if not isinstance(tr, str) or not tr.strip():
                    raise PromptCueRegistryError(
                        f'Entry "{defn.label}" contains an empty or non-string trigger.'
                    )

            for key, val in defn.routing_hints.items():
                if not isinstance(val, bool):
                    raise PromptCueRegistryError(
                        f'Entry "{defn.label}" routing_hint "{key}" must be a boolean.'
                    )

            for key, val in defn.action_hints.items():
                if not isinstance(val, bool):
                    raise PromptCueRegistryError(
                        f'Entry "{defn.label}" action_hint "{key}" must be a boolean.'
                    )
