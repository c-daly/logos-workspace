"""
Universal Edge - the connector of knowledge.

Edges represent relationships between nodes. Like nodes, edges
are uniform in structure - the type of relationship is captured
in the `relation` field, not in the class type.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

from .source import Source


@dataclass
class Edge:
    """
    A directed relationship between two nodes.

    The relation type is a string that emerges from usage, not
    from a predefined taxonomy. Common relations will naturally
    appear (is_a, causes, part_of, etc.) but new ones can be
    created at any time.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Connection
    source_id: str = ""  # The node this edge comes from
    target_id: str = ""  # The node this edge goes to
    relation: str = ""   # The type of relationship

    # Epistemic status
    confidence: float = 0.5
    source: Optional[Source] = None

    # Temporal validity (when is this relationship true?)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    # Quantitative attributes (optional)
    strength: Optional[float] = None   # For weighted associations (0-1)
    magnitude: Optional[float] = None  # For quantitative relations
    unit: Optional[str] = None         # Unit of magnitude

    # Flexible properties
    properties: dict[str, Any] = field(default_factory=dict)

    # Usage tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0

    def touch(self):
        """Record that this edge was accessed."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

    def is_temporal(self) -> bool:
        """Does this edge have bounded temporal validity?"""
        return self.valid_from is not None or self.valid_until is not None

    def is_currently_valid(self) -> bool:
        """Is this edge valid right now?"""
        now = datetime.now(timezone.utc)
        
        # Handle naive datetimes from old saved data
        valid_from = self.valid_from
        if valid_from and valid_from.tzinfo is None:
            valid_from = valid_from.replace(tzinfo=timezone.utc)
            
        valid_until = self.valid_until
        if valid_until and valid_until.tzinfo is None:
            valid_until = valid_until.replace(tzinfo=timezone.utc)
        
        if valid_from and now < valid_from:
            return False
        if valid_until and now > valid_until:
            return False
        return True

    def staleness_days(self) -> float:
        """How long since this was last accessed?"""
        accessed = self.last_accessed
        # Handle naive datetimes from old saved data
        if accessed.tzinfo is None:
            accessed = accessed.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - accessed
        return delta.total_seconds() / 86400

    def should_decay(self, decay_threshold: float = 0.1) -> bool:
        """
        Should this edge's strength decay due to disuse?

        Only applies to weighted associations.
        """
        if self.strength is None:
            return False
        return self.staleness_days() > 7 and self.strength > decay_threshold

    def decay_strength(self, rate: float = 0.95):
        """Apply decay to edge strength."""
        if self.strength is not None:
            self.strength *= rate

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "confidence": self.confidence,
            "source": self.source.to_dict() if self.source else None,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "strength": self.strength,
            "magnitude": self.magnitude,
            "unit": self.unit,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Edge":
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation=data["relation"],
            confidence=data.get("confidence", 0.5),
            source=Source.from_dict(data["source"]) if data.get("source") else None,
            valid_from=datetime.fromisoformat(data["valid_from"]) if data.get("valid_from") else None,
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None,
            strength=data.get("strength"),
            magnitude=data.get("magnitude"),
            unit=data.get("unit"),
            properties=data.get("properties", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else datetime.now(timezone.utc),
            access_count=data.get("access_count", 0),
        )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.id == other.id
        return False


# Common relation types (not exhaustive - new ones can be created)

class Relations:
    """Common relation types for reference."""

    # Structural
    IS_A = "is_a"                    # Type membership
    INSTANCE_OF = "instance_of"      # Specific instance
    PART_OF = "part_of"              # Compositional
    HAS_PART = "has_part"            # Inverse of part_of
    HAS_PROPERTY = "has_property"    # Attribute relationship

    # Causal
    CAUSES = "causes"                # Direct causation
    CAUSED_BY = "caused_by"          # Inverse
    ENABLES = "enables"              # Makes possible
    PREVENTS = "prevents"            # Blocks
    INFLUENCES = "influences"        # Weaker than causes

    # Temporal
    BEFORE = "before"                # Temporal ordering
    AFTER = "after"
    DURING = "during"                # Temporal containment
    STARTS = "starts"
    ENDS = "ends"

    # Logical
    IMPLIES = "implies"              # If A then B
    CONTRADICTS = "contradicts"      # Mutual exclusion
    EQUIVALENT_TO = "equivalent_to"  # Same thing
    GENERALIZES = "generalizes"      # More general than
    SPECIALIZES = "specializes"      # More specific than

    # Spatial
    LOCATED_AT = "located_at"
    CONTAINS = "contains"
    ADJACENT_TO = "adjacent_to"
    ABOVE = "above"
    BELOW = "below"

    # Quantitative
    EQUALS = "equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    APPROXIMATELY = "approximately"

    # Semantic
    SIMILAR_TO = "similar_to"
    DIFFERENT_FROM = "different_from"
    RELATED_TO = "related_to"
    EXAMPLE_OF = "example_of"

    # Meta/Epistemic
    DEFINED_AS = "defined_as"
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    CONTRADICTED_BY = "contradicted_by"
    LEARNED_FROM = "learned_from"


# Factory functions for common edge types

def create_is_a(
    source_id: str,
    target_id: str,
    source: Source = None,
    confidence: float = 0.8,
) -> Edge:
    """Create a type membership edge."""
    return Edge(
        source_id=source_id,
        target_id=target_id,
        relation=Relations.IS_A,
        source=source,
        confidence=confidence,
    )


def create_causes(
    source_id: str,
    target_id: str,
    source: Source = None,
    confidence: float = 0.6,
    strength: float = None,
) -> Edge:
    """Create a causal edge."""
    return Edge(
        source_id=source_id,
        target_id=target_id,
        relation=Relations.CAUSES,
        source=source,
        confidence=confidence,
        strength=strength,
    )


def create_association(
    source_id: str,
    target_id: str,
    relation: str = Relations.RELATED_TO,
    strength: float = 0.5,
    source: Source = None,
    confidence: float = 0.5,
) -> Edge:
    """Create a weighted association edge."""
    return Edge(
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        source=source,
        confidence=confidence,
        strength=strength,
    )


def create_quantitative(
    source_id: str,
    target_id: str,
    relation: str,
    magnitude: float,
    unit: str,
    source: Source = None,
    confidence: float = 0.7,
) -> Edge:
    """Create an edge with quantitative information."""
    return Edge(
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        source=source,
        confidence=confidence,
        magnitude=magnitude,
        unit=unit,
    )
