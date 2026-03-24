# promptcue | Pydantic models for PromptCue public schema
# Maintainer: Informity

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from promptcue.constants import (
    PCUE_BASIS_FALLBACK,
    PCUE_SCHEMA_VERSION,
    PCUE_UNKNOWN,
)
from promptcue.models.enums import PromptCueScope


class PromptCueCandidate(BaseModel):
    """A scored candidate query type from the classifier."""
    label: str
    score: float = Field(ge=0.0, le=1.0)
    basis: str   = PCUE_BASIS_FALLBACK


class PromptCueEntity(BaseModel):
    """A named entity extracted by spaCy — surface text plus entity type."""
    text:        str
    entity_type: str  # spaCy label: ORG, PRODUCT, GPE, PERSON, DATE, etc.


class PromptCueKeyword(BaseModel):
    """A keyword or keyphrase extracted by KeyBERT."""
    text:   str
    weight: float = Field(ge=0.0, le=1.0)
    kind:   str   = 'keyphrase'


class PromptCueLinguistics(BaseModel):
    """Linguistic features extracted from a query."""
    main_verbs:     list[str]             = Field(default_factory=list)
    noun_phrases:   list[str]             = Field(default_factory=list)
    named_entities: list[str]             = Field(default_factory=list)  # plain text, compat alias
    entities:       list[PromptCueEntity] = Field(default_factory=list)  # structured (text + type)

    @model_validator(mode='before')
    @classmethod
    def _sync_named_entities(cls, data: dict) -> dict:
        """Keep named_entities in sync with entities when only entities is provided."""
        if not data.get('named_entities') and data.get('entities'):
            data['named_entities'] = [
                e['text'] if isinstance(e, dict) else e.text
                for e in data['entities']
            ]
        return data


class PromptCueQueryObject(BaseModel):
    """Structured understanding of a single query — the public output of PromptCueAnalyzer."""

    # ==============================================================================
    # Identity and versioning
    # ==============================================================================
    schema_version:        str = PCUE_SCHEMA_VERSION
    input_text:            str
    normalized_text:       str
    language:              str

    # ==============================================================================
    # Classification
    # ==============================================================================
    primary_query_type:    str
    classification_basis:  str                 = PCUE_UNKNOWN
    candidate_query_types: list[PromptCueCandidate] = Field(default_factory=list)
    confidence:            float               = Field(ge=0.0, le=1.0)
    ambiguity_score:       float               = Field(ge=0.0, le=1.0)

    # ==============================================================================
    # Query dimensions
    # ==============================================================================
    scope: PromptCueScope = PromptCueScope.UNKNOWN

    # ==============================================================================
    # Linguistic enrichment (populated when enable_linguistic_extraction=True)
    # ==============================================================================
    main_verbs:     list[str]        = Field(default_factory=list)
    noun_phrases:   list[str]        = Field(default_factory=list)
    named_entities: list[str]             = Field(default_factory=list)  # plain text, compat alias
    entities:       list[PromptCueEntity] = Field(default_factory=list)  # structured (text + type)

    # ==============================================================================
    # Keyword enrichment (populated when enable_keyword_extraction=True)
    # ==============================================================================
    keywords: list[PromptCueKeyword] = Field(default_factory=list)

    # ==============================================================================
    # Routing and action directives
    # ==============================================================================
    routing_hints: dict[str, bool] = Field(default_factory=dict)
    action_hints:  dict[str, bool] = Field(default_factory=dict)

    # ==============================================================================
    # Constraints (reserved — populated in future milestones)
    # ==============================================================================
    constraints: list[str] = Field(default_factory=list)

    @property
    def query_type(self) -> str:
        """Compatibility alias for tests and simpler consumers."""
        return self.primary_query_type
