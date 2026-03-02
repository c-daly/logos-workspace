"""
Pydantic schemas for structured LLM output.

These models ensure type-safe, validated responses from LLM calls,
eliminating JSON parsing failures and silent field mismatches.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Extraction Output Schemas
# =============================================================================

class EntitySchema(BaseModel):
    """Schema for an extracted entity."""
    name: str = Field(..., description="Canonical name of the entity")
    temporary_id: str = Field(..., description="Temporary ID for reference in this extraction")
    properties: dict[str, Any] = Field(default_factory=dict, description="Key-value properties")
    temporal_grain: Literal[
        "instant", "seconds", "minutes", "hours", "days", 
        "months", "years", "decades", "geological", "eternal"
    ] = Field(default="eternal", description="Temporal granularity")
    claim_type: Literal[
        "fact", "definition", "rule", "association", "opinion", "hypothesis"
    ] = Field(default="fact", description="Type of knowledge claim")
    confidence_hint: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence 0-1")


class RelationSchema(BaseModel):
    """Schema for an extracted relation."""
    source: str = Field(..., description="Source entity temporary_id or existing node_id")
    relation: str = Field(..., description="Relation type (is_a, causes, part_of, etc.)")
    target: str = Field(..., description="Target entity temporary_id or existing node_id")
    properties: dict[str, Any] = Field(default_factory=dict, description="Relation properties")
    temporal_validity: Literal[
        "always", "sometimes", "past", "future", "conditional"
    ] = Field(default="always", description="When this relation holds")
    confidence_hint: float = Field(default=0.6, ge=0.0, le=1.0, description="Confidence 0-1")


class RuleSchema(BaseModel):
    """Schema for an extracted rule/pattern."""
    condition: str = Field(..., description="When this rule applies")
    consequence: str = Field(..., description="What follows from the condition")
    domain: str = Field(default="", description="Domain where this rule applies")
    confidence_hint: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence 0-1")


class UncertaintySchema(BaseModel):
    """Schema for something unclear or contradictory."""
    description: str = Field(..., description="What is unclear")
    related_entities: list[str] = Field(default_factory=list, description="Related entity IDs")


class MetaSchema(BaseModel):
    """Metadata about the extraction."""
    overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_notes: str = Field(default="", description="Any important context")


class ExtractionOutputSchema(BaseModel):
    """Complete structured output from extraction LLM call."""
    entities: list[EntitySchema] = Field(default_factory=list)
    relations: list[RelationSchema] = Field(default_factory=list)
    rules: list[RuleSchema] = Field(default_factory=list)
    uncertainties: list[UncertaintySchema] = Field(default_factory=list)
    meta: MetaSchema = Field(default_factory=MetaSchema)


# =============================================================================
# Critique Output Schemas
# =============================================================================

class OriginalRelationRef(BaseModel):
    """Reference to an original relation being reviewed."""
    source: str
    relation: str
    target: str


class NewEntitySchema(BaseModel):
    """A new entity suggested by the critic."""
    name: str
    temporary_id: str
    properties: dict[str, Any] = Field(default_factory=dict)


class NewRelationSchema(BaseModel):
    """A new relation suggested by the critic."""
    source: str
    relation: str
    target: str


class CorrectionSchema(BaseModel):
    """A correction with new entities and relations."""
    new_entities: list[NewEntitySchema] = Field(default_factory=list)
    new_relations: list[NewRelationSchema] = Field(default_factory=list)


class RelationReviewSchema(BaseModel):
    """Review of a single relation."""
    original: OriginalRelationRef
    verdict: Literal["approve", "reject", "correct"]
    reason: str = Field(default="", description="Explanation for the verdict")
    corrections: list[CorrectionSchema] = Field(default_factory=list)


class EntityReviewSchema(BaseModel):
    """Review of a single entity."""
    entity_id: str
    verdict: Literal["approve", "reject"]
    reason: str = Field(default="", description="Explanation for the verdict")


class CritiqueOutputSchema(BaseModel):
    """Complete structured output from critique LLM call."""
    reviews: list[RelationReviewSchema] = Field(default_factory=list)
    entity_reviews: list[EntityReviewSchema] = Field(default_factory=list)
    overall_notes: str = Field(default="", description="General observations")


# =============================================================================
# Reflection Output Schemas
# =============================================================================

class OrphanSchema(BaseModel):
    """An orphaned concept."""
    node_id: str
    suggestion: str = Field(default="", description="How to connect it")


class ContradictionSchema(BaseModel):
    """A contradiction between nodes."""
    nodes: list[str]
    description: str


class GapSchema(BaseModel):
    """A missing connection."""
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    missing_relation: str

    class Config:
        populate_by_name = True


class OvergeneralizationSchema(BaseModel):
    """A rule that might be too broad."""
    rule: str
    concern: str


class UndergeneralizationSchema(BaseModel):
    """Facts that might generalize."""
    facts: list[str]
    possible_rule: str


class ReflectionOutputSchema(BaseModel):
    """Complete structured output from reflection LLM call."""
    orphans: list[OrphanSchema] = Field(default_factory=list)
    contradictions: list[ContradictionSchema] = Field(default_factory=list)
    gaps: list[GapSchema] = Field(default_factory=list)
    overgeneralizations: list[OvergeneralizationSchema] = Field(default_factory=list)
    undergeneralizations: list[UndergeneralizationSchema] = Field(default_factory=list)


# =============================================================================
# Curiosity/Question Output Schemas
# =============================================================================

class QuestionSchema(BaseModel):
    """A generated question."""
    question: str
    motivation: str = Field(default="", description="Why this would help")
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class QuestionsOutputSchema(BaseModel):
    """Complete structured output from curiosity LLM call."""
    questions: list[QuestionSchema] = Field(default_factory=list)


# =============================================================================
# Hierarchy Inference Output Schemas
# =============================================================================

class InferredRelationSchema(BaseModel):
    """A relationship inferred between existing nodes."""
    source_name: str = Field(..., description="Name of the child/specific node")
    target_name: str = Field(..., description="Name of the parent/general node")
    relation: Literal["is_a", "part_of", "subtype_of", "instance_of"] = Field(
        default="is_a", description="Type of hierarchical relationship"
    )
    confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="How confident")
    reasoning: str = Field(default="", description="Why this relationship exists")


class HierarchyInferenceSchema(BaseModel):
    """Output from hierarchy inference LLM call."""
    inferred_relations: list[InferredRelationSchema] = Field(default_factory=list)
    notes: str = Field(default="", description="Any observations about the graph structure")


# =============================================================================
# Polysemy Split Output Schemas
# =============================================================================

class DomainClusterSchema(BaseModel):
    """A cluster of edges belonging to a single domain/sense."""
    domain_label: str = Field(..., description="Short domain qualifier, e.g., 'mathematics', 'psychology'")
    edge_indices: list[int] = Field(..., description="Indices of edges belonging to this domain")
    reasoning: str = Field(default="", description="Why these edges form a coherent domain")


class PolysemySplitSchema(BaseModel):
    """Output from polysemy detection LLM call."""
    should_split: bool = Field(..., description="Whether this node represents multiple distinct concepts")
    clusters: list[DomainClusterSchema] = Field(default_factory=list, description="Domain clusters if splitting")
    reasoning: str = Field(default="", description="Overall reasoning for the decision")


class SemanticDimensionsSchema(BaseModel):
    """Output from semantic dimensions LLM call."""
    dimensions: list[str] = Field(
        ..., 
        description="List of distinct semantic domains/meanings this word can have (e.g., ['mathematics', 'psychology'] for 'closure')"
    )
    is_polysemous: bool = Field(
        ..., 
        description="True if the word has multiple genuinely different meanings, not just different applications"
    )



# =============================================================================
# Introspection Output Schemas
# =============================================================================

class PerceptionSchema(BaseModel):
    """A perception the agent has about the user or situation."""
    content: str = Field(..., description="Description of what is perceived")
    about: Literal["user", "situation", "self"] = Field(
        default="user", description="What the perception is about"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in perception")
    temporal_grain: Literal[
        "instant", "seconds", "minutes", "hours", "days", 
        "months", "years", "decades", "geological", "eternal"
    ] = Field(default="minutes", description="How long this perception is relevant")


class ThoughtSchema(BaseModel):
    """A thought the agent has in response to the interaction."""
    content: str = Field(..., description="The thought itself")
    about: list[str] = Field(default_factory=list, description="Concepts this thought relates to")
    thought_type: Literal[
        "connection", "confusion", "curiosity", "realization", "question"
    ] = Field(default="connection", description="Type of thought")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in thought")
    temporal_grain: Literal[
        "instant", "seconds", "minutes", "hours", "days", 
        "months", "years", "decades", "geological", "eternal"
    ] = Field(default="minutes", description="How long this thought is relevant")


class FeelingSchema(BaseModel):
    """A feeling or internal state the agent experiences."""
    content: str = Field(..., description="Description of the feeling")
    valence: Literal["positive", "negative", "neutral"] = Field(
        default="neutral", description="Emotional valence"
    )
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="Intensity of feeling")
    temporal_grain: Literal[
        "instant", "seconds", "minutes", "hours", "days", 
        "months", "years", "decades", "geological", "eternal"
    ] = Field(default="minutes", description="How long this feeling persists")


class IntrospectionOutputSchema(BaseModel):
    """Output from introspection LLM call."""
    perceptions: list[PerceptionSchema] = Field(
        default_factory=list, description="Perceptions about user/situation"
    )
    thoughts: list[ThoughtSchema] = Field(
        default_factory=list, description="Thoughts in response to interaction"
    )
    feelings: list[FeelingSchema] = Field(
        default_factory=list, description="Internal feelings/states"
    )
