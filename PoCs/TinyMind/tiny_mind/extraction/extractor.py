"""
Knowledge Extractor

Uses an LLM to extract structured knowledge from natural language,
then creates nodes and edges that can be integrated into the knowledge graph.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Optional

from ..substrate.node import Node, TemporalGrain, GroundingType


# Map LLM temporal grain responses to valid enum values
TEMPORAL_GRAIN_MAP = {
    'instant': 'instant',
    'milliseconds': 'instant',
    'seconds': 'seconds',
    'second': 'seconds',
    'minutes': 'minutes',
    'minute': 'minutes',
    'hours': 'hours',
    'hour': 'hours',
    'days': 'days',
    'day': 'days',
    'week': 'days',
    'weeks': 'days',
    'months': 'months',
    'month': 'months',
    'years': 'years',
    'year': 'years',
    'decades': 'decades',
    'decade': 'decades',
    'centuries': 'decades',
    'geological': 'geological',
    'eternal': 'eternal',
    'timeless': 'eternal',
    'always': 'eternal',
}
from ..substrate.edge import Edge
from ..substrate.source import Source, SourceType
from ..substrate.graph import KnowledgeGraph

from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    CONTEXT_TEMPLATE,
    USER_INPUT_TEMPLATE,
    REFLECTION_PROMPT,
    CURIOSITY_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    CRITIC_INPUT_TEMPLATE,
    INTROSPECTION_SYSTEM_PROMPT,
    INTROSPECTION_INPUT_TEMPLATE,
)

from .schemas import (
    ExtractionOutputSchema,
    CritiqueOutputSchema,
    ReflectionOutputSchema,
    QuestionsOutputSchema,
    IntrospectionOutputSchema,
)


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""
    name: str
    temporary_id: str
    properties: dict = field(default_factory=dict)
    temporal_grain: str = "eternal"
    claim_type: str = "fact"
    confidence_hint: float = 0.7


@dataclass
class ExtractedRelation:
    """A relation extracted from text."""
    source: str  # temporary_id or existing node_id
    relation: str
    target: str
    properties: dict = field(default_factory=dict)
    temporal_validity: str = "always"
    confidence_hint: float = 0.6


@dataclass
class ExtractedRule:
    """A rule/pattern extracted from text."""
    condition: str
    consequence: str
    domain: str = ""
    confidence_hint: float = 0.5


@dataclass
class Uncertainty:
    """Something unclear or contradictory."""
    description: str
    related_entities: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Complete result of an extraction."""
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    rules: list[ExtractedRule] = field(default_factory=list)
    uncertainties: list[Uncertainty] = field(default_factory=list)
    overall_confidence: float = 0.5
    extraction_notes: str = ""
    raw_response: str = ""


@dataclass
class CritiqueResult:
    """Result of critiquing an extraction."""
    approved_relations: list[ExtractedRelation] = field(default_factory=list)
    rejected_relations: list[dict] = field(default_factory=list)  # {relation, reason}
    corrected_relations: list[ExtractedRelation] = field(default_factory=list)
    approved_entities: list[ExtractedEntity] = field(default_factory=list)
    suggested_entities: list[ExtractedEntity] = field(default_factory=list)  # New entities from corrections
    critique_notes: str = ""


class Extractor:
    """
    Extracts structured knowledge from natural language using an LLM.

    Can use OpenAI, Anthropic, or a local model via Hermes.
    Supports separate models for extraction and critique to reduce shared biases.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = None,
        api_key: str = None,
        hermes_url: str = None,
        # Critic configuration - uses different model to avoid shared biases
        critic_provider: str = None,  # Defaults to same as llm_provider
        critic_model: str = None,  # Defaults to same as model
        critic_api_key: str = None,  # Defaults to same as api_key
    ):
        self.llm_provider = llm_provider
        self.model = model or self._default_model(llm_provider)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.hermes_url = hermes_url or os.environ.get("HERMES_URL", "http://localhost:18000")

        # Critic can use a different provider/model for diverse perspectives
        self.critic_provider = critic_provider or llm_provider
        self.critic_model = critic_model or self._default_model(self.critic_provider)
        self.critic_api_key = critic_api_key or self._get_api_key(self.critic_provider)

        self._client = None
        self._critic_client = None

    def _get_api_key(self, provider: str) -> str:
        """Get API key for a provider."""
        if provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        elif provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        return None

    def _default_model(self, provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "hermes": "default",
        }
        return defaults.get(provider, "gpt-4o-mini")

    def _get_client(self, for_critic: bool = False):
        """Lazy-load the LLM client."""
        if for_critic:
            if self._critic_client is not None:
                return self._critic_client
            provider = self.critic_provider
            api_key = self.critic_api_key
        else:
            if self._client is not None:
                return self._client
            provider = self.llm_provider
            api_key = self.api_key

        if provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package required. pip install openai")

        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package required. pip install anthropic")

        elif provider == "hermes":
            import httpx
            client = httpx.Client(base_url=self.hermes_url)

        else:
            raise ValueError(f"Unknown provider: {provider}")

        if for_critic:
            self._critic_client = client
        else:
            self._client = client

        return client

    def _call_llm(self, system_prompt: str, user_prompt: str, for_critic: bool = False) -> str:
        """Call the LLM and get a response."""
        client = self._get_client(for_critic=for_critic)
        provider = self.critic_provider if for_critic else self.llm_provider
        model = self.critic_model if for_critic else self.model

        if provider == "openai":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent extraction
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        elif provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        elif provider == "hermes":
            response = client.post("/llm", json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
            })
            return response.json().get("content", "{}")

    def _call_llm_structured(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        response_schema: type,
        for_critic: bool = False
    ):
        """
        Call the LLM with structured output validation.
        
        Uses OpenAI's structured output feature to guarantee schema compliance.
        Falls back to regular JSON parsing for non-OpenAI providers.
        
        Args:
            system_prompt: System message for the LLM
            user_prompt: User message for the LLM
            response_schema: Pydantic model class for response validation
            for_critic: Whether to use the critic model
            
        Returns:
            Validated instance of response_schema
        """
        from pydantic import ValidationError
        
        client = self._get_client(for_critic=for_critic)
        provider = self.critic_provider if for_critic else self.llm_provider
        model = self.critic_model if for_critic else self.model

        if provider == "openai":
            # Use OpenAI's native structured output
            try:
                response = client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    response_format=response_schema,
                )
                return response.choices[0].message.parsed
            except Exception as e:
                # Fall back to regular JSON mode if structured output fails
                raw = self._call_llm(system_prompt, user_prompt, for_critic)
                return response_schema.model_validate_json(raw)

        elif provider == "anthropic":
            # Anthropic doesn't have native structured output yet
            # Use regular call + Pydantic validation
            raw = self._call_llm(system_prompt, user_prompt, for_critic)
            try:
                return response_schema.model_validate_json(raw)
            except ValidationError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    return response_schema.model_validate_json(json_match.group())
                raise

        else:
            # Hermes/other providers - use Pydantic validation on response
            raw = self._call_llm(system_prompt, user_prompt, for_critic)
            return response_schema.model_validate_json(raw)

    def _format_existing_nodes(self, graph: KnowledgeGraph, limit: int = 50) -> str:
        """Format existing nodes for context."""
        nodes = list(graph.nodes())[:limit]
        if not nodes:
            return "(no existing knowledge)"

        lines = []
        for node in nodes:
            props = ", ".join(f"{k}={v}" for k, v in list(node.properties.items())[:3])
            lines.append(f"- {node.id}: {node.name} ({props})")

        return "\n".join(lines)

    def _find_similar_node(
        self, 
        name: str, 
        graph: "KnowledgeGraph",
        threshold: float = 0.85
    ) -> Optional[Node]:
        """
        Find an existing node with similar name using trigram index.
        
        Uses the graph's find_similar_nodes for O(k) lookup instead of O(n) scan.
        """
        name_lower = name.lower().strip()
        
        # 1. Exact match first (O(1) via index)
        existing = graph.find_node(name)
        if existing:
            return existing
        
        # 2. Check plural/singular variants (common cases)
        for variant in self._plural_variants(name_lower):
            existing = graph.find_node(variant)
            if existing:
                return existing
        
        # 3. Use trigram index for fuzzy matching
        similar = graph.find_similar_nodes(name, threshold=threshold, limit=1)
        if similar:
            return similar[0][0]  # Return the node (first element of tuple)
        
        return None
    
    def _plural_variants(self, name: str) -> list[str]:
        """Generate common plural/singular variants of a name."""
        variants = []
        name_lower = name.lower()
        
        # Singular -> Plural
        if name_lower.endswith('y') and len(name_lower) > 2:
            variants.append(name_lower[:-1] + 'ies')
        elif name_lower.endswith(('s', 'x', 'z', 'ch', 'sh')):
            variants.append(name_lower + 'es')
        else:
            variants.append(name_lower + 's')
        
        # Plural -> Singular
        if name_lower.endswith('ies') and len(name_lower) > 4:
            variants.append(name_lower[:-3] + 'y')
        elif name_lower.endswith('es') and len(name_lower) > 3:
            variants.append(name_lower[:-2])
        elif name_lower.endswith('s') and len(name_lower) > 2:
            variants.append(name_lower[:-1])
        
        return variants
    
    def _is_plural_variant(self, a: str, b: str) -> bool:
        """Check if two strings are singular/plural variants."""
        # Simple pluralization rules
        if a + "s" == b or b + "s" == a:
            return True
        if a + "es" == b or b + "es" == a:
            return True
        # cities/city (endswith, not rstrip!)
        if a.endswith("ies") and a[:-3] + "y" == b:
            return True
        if b.endswith("ies") and b[:-3] + "y" == a:
            return True
        # matrices/matrix
        if a.endswith("ices") and a[:-4] + "ix" == b:
            return True
        if b.endswith("ices") and b[:-4] + "ix" == a:
            return True
        return False
    
    def _find_similar_rule(
        self,
        condition: str,
        consequence: str,
        graph: "KnowledgeGraph",
        threshold: float = 0.90
    ) -> Optional[Node]:
        """
        Find an existing rule node with similar condition/consequence.
        
        Rules need high similarity since they encode specific logic.
        """
        cond_lower = condition.lower().strip()
        cons_lower = consequence.lower().strip()
        
        for node in graph.nodes():
            # Only check rule nodes
            if not node.properties.get("is_rule"):
                continue
            
            existing_cond = node.properties.get("condition", "").lower().strip()
            existing_cons = node.properties.get("consequence", "").lower().strip()
            
            # Check condition similarity
            cond_score = SequenceMatcher(None, cond_lower, existing_cond).ratio()
            cons_score = SequenceMatcher(None, cons_lower, existing_cons).ratio()
            
            # Both condition and consequence must be similar
            if cond_score >= threshold and cons_score >= threshold:
                return node
            
            # Or exact condition match with similar consequence
            if cond_lower == existing_cond and cons_score >= 0.8:
                return node
        
        return None

    def extract(
        self,
        text: str,
        graph: KnowledgeGraph = None,
    ) -> ExtractionResult:
        """
        Extract knowledge from text using structured LLM output.

        Args:
            text: The text to extract from
            graph: Optional existing graph for context

        Returns:
            ExtractionResult with extracted entities, relations, rules
        """
        # Build context
        context = ""
        if graph:
            existing_nodes = self._format_existing_nodes(graph)
            context = CONTEXT_TEMPLATE.format(existing_nodes=existing_nodes)

        # Build user prompt
        user_prompt = context + "\n" + USER_INPUT_TEMPLATE.format(user_input=text)

        # Call LLM with structured output
        try:
            data = self._call_llm_structured(
                EXTRACTION_SYSTEM_PROMPT, 
                user_prompt, 
                ExtractionOutputSchema
            )
        except Exception as e:
            return ExtractionResult(
                extraction_notes=f"Failed to get structured response: {e}",
            )

        # Build result from validated schema
        result = ExtractionResult()

        # Convert schema objects to dataclass objects
        for ent in data.entities:
            result.entities.append(ExtractedEntity(
                name=ent.name,
                temporary_id=ent.temporary_id,
                properties=ent.properties,
                temporal_grain=ent.temporal_grain,
                claim_type=ent.claim_type,
                confidence_hint=ent.confidence_hint,
            ))

        for rel in data.relations:
            result.relations.append(ExtractedRelation(
                source=rel.source,
                relation=rel.relation,
                target=rel.target,
                properties=rel.properties,
                temporal_validity=rel.temporal_validity,
                confidence_hint=rel.confidence_hint,
            ))

        for rule in data.rules:
            result.rules.append(ExtractedRule(
                condition=rule.condition,
                consequence=rule.consequence,
                domain=rule.domain,
                confidence_hint=rule.confidence_hint,
            ))

        for unc in data.uncertainties:
            result.uncertainties.append(Uncertainty(
                description=unc.description,
                related_entities=unc.related_entities,
            ))

        # Meta
        result.overall_confidence = data.meta.overall_confidence
        result.extraction_notes = data.meta.extraction_notes

        return result

    def critique(
        self,
        result: ExtractionResult,
        original_text: str,
    ) -> CritiqueResult:
        """
        Critique an extraction result using structured LLM output.

        Reviews each relation for semantic accuracy, flags absurd connections,
        and suggests corrections for compressed causal chains.

        Args:
            result: The extraction to critique
            original_text: The original text that was extracted from

        Returns:
            CritiqueResult with approved, rejected, and corrected items
        """
        # Format entities for the critic
        entities_str = json.dumps([
            {
                "temporary_id": e.temporary_id,
                "name": e.name,
                "properties": e.properties,
            }
            for e in result.entities
        ], indent=2)

        # Format relations for the critic
        relations_str = json.dumps([
            {
                "source": r.source,
                "relation": r.relation,
                "target": r.target,
                "confidence_hint": r.confidence_hint,
            }
            for r in result.relations
        ], indent=2)

        # Build the critic prompt
        user_prompt = CRITIC_INPUT_TEMPLATE.format(
            original_text=original_text,
            entities=entities_str,
            relations=relations_str,
        )

        # Call the critic with structured output
        try:
            data = self._call_llm_structured(
                CRITIC_SYSTEM_PROMPT, 
                user_prompt, 
                CritiqueOutputSchema,
                for_critic=True
            )
        except Exception as e:
            # If structured output fails, fail-closed (reject relations, keep entities)
            return CritiqueResult(
                approved_entities=result.entities,
                rejected_relations=[
                    {"relation": r, "reason": f"Critique failed: {e}"}
                    for r in result.relations
                ],
                critique_notes=f"Structured critique failed: {e}",
            )

        # Build critique result from validated schema
        critique = CritiqueResult()
        critique.critique_notes = data.overall_notes

        # Build a map of original relations for matching
        original_relations = {
            (r.source, r.relation, r.target): r
            for r in result.relations
        }

        # Process relation reviews
        for review in data.reviews:
            key = (review.original.source, review.original.relation, review.original.target)

            if key in original_relations:
                original_rel = original_relations[key]

                if review.verdict == "approve":
                    critique.approved_relations.append(original_rel)
                elif review.verdict == "reject":
                    critique.rejected_relations.append({
                        "relation": original_rel,
                        "reason": review.reason or "No reason given",
                    })
                elif review.verdict == "correct":
                    # Process corrections
                    for correction in review.corrections:
                        # Add new entities
                        for new_ent in correction.new_entities:
                            critique.suggested_entities.append(ExtractedEntity(
                                name=new_ent.name,
                                temporary_id=new_ent.temporary_id,
                                properties=new_ent.properties,
                                confidence_hint=0.5,
                            ))
                        # Add corrected relations
                        for new_rel in correction.new_relations:
                            critique.corrected_relations.append(ExtractedRelation(
                                source=new_rel.source,
                                relation=new_rel.relation,
                                target=new_rel.target,
                                confidence_hint=original_rel.confidence_hint * 0.9,
                            ))

        # Process entity reviews
        entity_map = {e.temporary_id: e for e in result.entities}
        reviewed_ids = set()
        
        for review in data.entity_reviews:
            reviewed_ids.add(review.entity_id)
            if review.entity_id in entity_map:
                if review.verdict == "approve":
                    critique.approved_entities.append(entity_map[review.entity_id])
            # Rejected entities are simply not added

        # Any entities not explicitly reviewed are approved
        for entity in result.entities:
            if entity.temporary_id not in reviewed_ids:
                if entity not in critique.approved_entities:
                    critique.approved_entities.append(entity)

        return critique

    def extract_and_critique(
        self,
        text: str,
        graph: KnowledgeGraph = None,
    ) -> tuple[ExtractionResult, CritiqueResult]:
        """
        Extract knowledge and critique it in one call.

        Convenience method that runs extraction then critique.
        Returns both results so caller can inspect what was changed.
        """
        extraction = self.extract(text, graph)
        critique = self.critique(extraction, text)
        return extraction, critique

    def integrate(
        self,
        result: ExtractionResult,
        graph: KnowledgeGraph,
        source_text: str,
    ) -> tuple[list[Node], list[Edge]]:
        """
        Integrate extracted knowledge into the graph.

        Returns lists of created nodes and edges.
        """
        created_nodes = []
        created_edges = []

        # Map temporary IDs to real node IDs
        temp_to_real: dict[str, str] = {}

        # Create source record
        source = Source.from_llm(source_text)

        # Create nodes for entities
        for entity in result.entities:
            # Check if this matches an existing node (with fuzzy matching)
            existing = self._find_similar_node(entity.name, graph)
            if existing:
                temp_to_real[entity.temporary_id] = existing.id
                
                # CORROBORATION: Seeing the same entity again increases confidence
                # Bayesian-ish update: confidence approaches 1.0 with diminishing returns
                old_confidence = existing.confidence
                existing.confidence = old_confidence + (1 - old_confidence) * 0.1
                
                # Record confirmation on source if available
                if existing.source:
                    existing.source.record_confirmation()
                
                # Update existing node with new properties (refinement)
                for k, v in entity.properties.items():
                    if k not in existing.properties:
                        existing.properties[k] = v
                    elif k == "definition" and existing.properties[k] != v:
                        # Append additional definitions
                        existing.properties[k] += f" Also: {v}"
            else:
                # Map temporal grain to valid enum value
                grain_str = TEMPORAL_GRAIN_MAP.get(
                    entity.temporal_grain.lower(),
                    'eternal'  # default if unknown
                )
                # Create new node - start with LOW confidence
                # Confidence builds through corroboration, not initial claims
                initial_confidence = min(
                    entity.confidence_hint * source.reliability * 0.5,  # Halve it
                    0.5  # Cap at 0.5 for new uncorroborated claims
                )
                node = Node(
                    name=entity.name,
                    properties=entity.properties,
                    source=source,
                    confidence=initial_confidence,
                    temporal_grain=TemporalGrain(grain_str),
                )
                graph.add_node(node)
                temp_to_real[entity.temporary_id] = node.id
                created_nodes.append(node)

        # Create edges for relations
        for relation in result.relations:
            # Resolve source and target
            source_id = temp_to_real.get(relation.source, relation.source)
            target_id = temp_to_real.get(relation.target, relation.target)

            # Auto-create nodes for undefined temporary IDs (LLM mentioned them in relations)
            for temp_id, real_id in [(relation.source, source_id), (relation.target, target_id)]:
                if real_id not in graph._nodes and temp_id.endswith(('_001', '_002', '_003', '_process', '_event')):
                    # Extract a name from the temp_id (e.g., "falling_001" -> "falling")
                    inferred_name = temp_id.rsplit('_', 1)[0]
                    
                    # Check if a similar node already exists
                    existing = self._find_similar_node(inferred_name, graph)
                    if existing:
                        temp_to_real[temp_id] = existing.id
                        if temp_id == relation.source:
                            source_id = existing.id
                        else:
                            target_id = existing.id
                        continue
                    
                    node = Node(
                        name=inferred_name,
                        properties={'auto_created': True, 'from_relation': True},
                        source=source,
                        confidence=0.4,  # Lower confidence for auto-created
                    )
                    graph.add_node(node)
                    temp_to_real[temp_id] = node.id
                    created_nodes.append(node)
                    # Update the ID we're using
                    if temp_id == relation.source:
                        source_id = node.id
                    else:
                        target_id = node.id

            # Check both nodes exist
            if source_id not in graph._nodes or target_id not in graph._nodes:
                continue

            # Check if this relation already exists (corroboration)
            existing_edges = graph.find_edges(
                source_id=source_id,
                target_id=target_id,
                relation=relation.relation,
            )
            
            if existing_edges:
                # CORROBORATION: Same relation seen again - boost confidence
                existing_edge = existing_edges[0]
                old_confidence = existing_edge.confidence
                existing_edge.confidence = old_confidence + (1 - old_confidence) * 0.1
                if existing_edge.source:
                    existing_edge.source.record_confirmation()
                # Merge any new properties
                for k, v in relation.properties.items():
                    if k not in existing_edge.properties:
                        existing_edge.properties[k] = v
            else:
                # Create new edge
                edge = Edge(
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation.relation,
                    source=source,
                    confidence=relation.confidence_hint * source.reliability,
                    properties=relation.properties,
                )
                graph.add_edge(edge)
                created_edges.append(edge)

        # Handle rules by creating special rule nodes
        for rule in result.rules:
            # Check if a similar rule already exists
            existing_rule = self._find_similar_rule(
                rule.condition, rule.consequence, graph
            )
            if existing_rule:
                # CORROBORATION: Boost confidence of existing rule
                old_confidence = existing_rule.confidence
                existing_rule.confidence = old_confidence + (1 - old_confidence) * 0.1
                if existing_rule.source:
                    existing_rule.source.record_confirmation()
                # Update domain if more specific
                if rule.domain and not existing_rule.properties.get("domain"):
                    existing_rule.properties["domain"] = rule.domain
                continue
            
            rule_node = Node(
                name=f"rule: {rule.condition[:30]}...",
                properties={
                    "is_rule": True,
                    "condition": rule.condition,
                    "consequence": rule.consequence,
                    "domain": rule.domain,
                },
                source=source,
                confidence=rule.confidence_hint * source.reliability,
                temporal_grain=TemporalGrain.ETERNAL,
            )
            graph.add_node(rule_node)
            created_nodes.append(rule_node)

        return created_nodes, created_edges

    def integrate_critiqued(
        self,
        critique: CritiqueResult,
        graph: KnowledgeGraph,
        source_text: str,
    ) -> tuple[list[Node], list[Edge], list[dict]]:
        """
        Integrate knowledge that has passed critique.

        Uses approved entities/relations and critic's corrections.
        Returns (created_nodes, created_edges, rejected_relations).
        """
        created_nodes = []
        created_edges = []

        # Map temporary IDs to real node IDs
        temp_to_real: dict[str, str] = {}

        # Create source record
        source = Source.from_llm(source_text)

        # Process approved entities + suggested entities from corrections
        all_entities = critique.approved_entities + critique.suggested_entities
        for entity in all_entities:
            existing = graph.find_node(entity.name)
            if existing:
                temp_to_real[entity.temporary_id] = existing.id
                for k, v in entity.properties.items():
                    if k not in existing.properties:
                        existing.properties[k] = v
            else:
                grain_str = TEMPORAL_GRAIN_MAP.get(
                    entity.temporal_grain.lower() if hasattr(entity, 'temporal_grain') else 'eternal',
                    'eternal'
                )
                initial_confidence = min(
                    entity.confidence_hint * source.reliability * 0.5,
                    0.5
                )
                node = Node(
                    name=entity.name,
                    properties=entity.properties,
                    source=source,
                    confidence=initial_confidence,
                    temporal_grain=TemporalGrain(grain_str),
                )
                graph.add_node(node)
                temp_to_real[entity.temporary_id] = node.id
                created_nodes.append(node)

        # Process approved relations + corrected relations
        all_relations = critique.approved_relations + critique.corrected_relations
        for relation in all_relations:
            source_id = temp_to_real.get(relation.source, relation.source)
            target_id = temp_to_real.get(relation.target, relation.target)

            # Auto-create nodes for undefined temporary IDs
            for temp_id, real_id in [(relation.source, source_id), (relation.target, target_id)]:
                if real_id not in graph._nodes and temp_id.endswith(('_001', '_002', '_003', '_process', '_event')):
                    inferred_name = temp_id.rsplit('_', 1)[0]
                    node = Node(
                        name=inferred_name,
                        properties={'auto_created': True, 'from_relation': True, 'from_critic': True},
                        source=source,
                        confidence=0.4,
                    )
                    graph.add_node(node)
                    temp_to_real[temp_id] = node.id
                    created_nodes.append(node)
                    if temp_id == relation.source:
                        source_id = node.id
                    else:
                        target_id = node.id

            if source_id not in graph._nodes or target_id not in graph._nodes:
                continue

            edge = Edge(
                source_id=source_id,
                target_id=target_id,
                relation=relation.relation,
                source=source,
                confidence=relation.confidence_hint * source.reliability,
                properties=relation.properties,
            )
            graph.add_edge(edge)
            created_edges.append(edge)

        return created_nodes, created_edges, critique.rejected_relations

    def reflect(self, graph: KnowledgeGraph) -> dict:
        """
        Reflect on the current state of knowledge using structured output.

        Returns analysis of orphans, contradictions, gaps, etc.
        """
        # Build graph summary
        stats = graph.get_stats()
        orphans = graph.get_orphans()
        top_nodes = graph.get_highly_connected(10)

        summary = f"""
Nodes: {stats.node_count}
Edges: {stats.edge_count}
Orphans: {stats.orphan_count}
Avg confidence: {stats.avg_confidence:.2f}

Most connected: {', '.join(n.name for n, _ in top_nodes[:5])}

Orphan nodes: {', '.join(n.name for n in orphans[:10])}
"""

        user_prompt = REFLECTION_PROMPT.format(graph_summary=summary)
        
        try:
            result = self._call_llm_structured(
                EXTRACTION_SYSTEM_PROMPT, 
                user_prompt,
                ReflectionOutputSchema
            )
            return result.model_dump()
        except Exception as e:
            return {"error": f"Failed to get structured reflection: {e}"}

    def generate_questions(self, graph: KnowledgeGraph, recent_topics: list[str] = None) -> list[dict]:
        """
        Generate questions that would help the agent learn more.
        
        Uses structured output to guarantee valid question format.
        """
        stats = graph.get_stats()
        orphans = graph.get_orphans()

        summary = f"""
Nodes: {stats.node_count}
Orphans: {[n.name for n in orphans[:5]]}
Recent accesses: {[n.name for n in list(graph.nodes()) if n.access_count > 0][:5]}
"""

        topics_str = ", ".join(recent_topics or ["none yet"])

        user_prompt = CURIOSITY_PROMPT.format(
            graph_summary=summary,
            recent_topics=topics_str,
        )

        try:
            result = self._call_llm_structured(
                EXTRACTION_SYSTEM_PROMPT, 
                user_prompt,
                QuestionsOutputSchema
            )
            return [q.model_dump() for q in result.questions]
        except Exception:
            return []


    def introspect(
        self,
        user_input: str,
        extraction_summary: str,
        existing_context: str,
        history: str,
    ) -> dict:
        """
        Generate the agent's internal mental response to an interaction.
        
        Returns structured introspection with perceptions, thoughts, and feelings.
        
        Args:
            user_input: What the user said
            extraction_summary: Summary of what knowledge was extracted
            existing_context: Relevant existing knowledge
            history: Recent conversation history
            
        Returns:
            Dict with perceptions, thoughts, feelings lists
        """
        user_prompt = INTROSPECTION_INPUT_TEMPLATE.format(
            user_input=user_input,
            extraction_summary=extraction_summary,
            existing_context=existing_context,
            history=history,
        )
        
        try:
            result = self._call_llm_structured(
                INTROSPECTION_SYSTEM_PROMPT,
                user_prompt,
                IntrospectionOutputSchema,
            )
            return result.model_dump()
        except Exception as e:
            # Introspection is not critical - fail gracefully
            return {"perceptions": [], "thoughts": [], "feelings": [], "error": str(e)}
