"""
Universal Node - the atom of knowledge.

A node can represent anything: a physical object, an abstract concept,
a mathematical operator, an event, an idea, a theory. The "type" emerges
from patterns of connection, not from pre-assigned categories.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid

from .source import Source


class TemporalGrain(Enum):
    """
    The natural timescale at which something exists or matters.

    This helps the system reason about relevance: when thinking about
    "will this cup break?", geological timescales aren't relevant.
    """

    INSTANT = "instant"          # < 100ms (collision, click, flash)
    SECONDS = "seconds"          # 100ms - 60s (falling, utterance)
    MINUTES = "minutes"          # 1-60 min (conversation, cooking)
    HOURS = "hours"              # 1-24 hours (workday, flight)
    DAYS = "days"                # 1-30 days (project, trip)
    MONTHS = "months"            # 1-12 months (semester, season)
    YEARS = "years"              # 1-100 years (career, lifetime)
    DECADES = "decades"          # 10-100 years (historical period)
    GEOLOGICAL = "geological"    # 1000+ years (evolution, climate)
    ETERNAL = "eternal"          # Timeless truths (math, logic)


class GroundingType(Enum):
    """
    How is this concept connected to perception/action?

    Grounded concepts have direct sensory or motor experience.
    Ungrounded concepts are purely abstract/symbolic.
    """

    UNGROUNDED = "ungrounded"    # Pure abstraction (numbers, logic)
    PERCEPTUAL = "perceptual"    # Seen/heard/sensed
    MOTOR = "motor"              # Can act on / manipulate
    BOTH = "both"                # Both perceived and acted upon


@dataclass
class Node:
    """
    The universal atom of knowledge.

    Everything is a node. The "what kind of thing is this?" question
    is answered by looking at its properties and connections, not
    by checking a type field.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""  # Human-readable label (can change as understanding grows)

    # Semantic position (populated by embedding model)
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None

    # Epistemic status
    confidence: float = 0.5  # How sure are we this exists/is true?
    source: Optional[Source] = None

    # Temporal characteristics
    temporal_grain: TemporalGrain = TemporalGrain.ETERNAL
    valid_from: Optional[datetime] = None   # When does this become true?
    valid_until: Optional[datetime] = None  # When does this stop being true?

    # Grounding in perception/action
    grounding_type: GroundingType = GroundingType.UNGROUNDED
    grounding_refs: list[str] = field(default_factory=list)  # Links to sensory data

    # Flexible properties - anything else we know about this
    properties: dict[str, Any] = field(default_factory=dict)

    # Usage tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0

    def touch(self):
        """Record that this node was accessed."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

    def set_property(self, key: str, value: Any):
        """Set a property, handling special cases."""
        self.properties[key] = value
        self.touch()

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value (does not update access tracking)."""
        return self.properties.get(key, default)

    def has_property(self, key: str) -> bool:
        """Check if node has a property."""
        return key in self.properties

    def is_temporal(self) -> bool:
        """Does this node have bounded temporal validity?"""
        return self.valid_from is not None or self.valid_until is not None

    def is_eternal(self) -> bool:
        """Is this a timeless truth?"""
        return self.temporal_grain == TemporalGrain.ETERNAL and not self.is_temporal()

    def is_grounded(self) -> bool:
        """Is this connected to perception/action?"""
        return self.grounding_type != GroundingType.UNGROUNDED

    def age_days(self) -> float:
        """How old is this knowledge in days?"""
        created = self.created_at
        # Handle naive datetimes from old saved data
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - created
        return delta.total_seconds() / 86400

    def staleness_days(self) -> float:
        """How long since this was last accessed?"""
        accessed = self.last_accessed
        # Handle naive datetimes from old saved data
        if accessed.tzinfo is None:
            accessed = accessed.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - accessed
        return delta.total_seconds() / 86400

    def should_prune(self, max_staleness_days: float = 30, min_access_count: int = 2) -> bool:
        """
        Should this node be pruned due to disuse?

        Eternal truths and high-confidence nodes are protected.
        """
        if self.is_eternal() and self.confidence > 0.7:
            return False
        if self.access_count >= min_access_count:
            return False
        return self.staleness_days() > max_staleness_days

    def merge_properties(self, other: "Node"):
        """
        Merge properties from another node (e.g., when combining duplicates).
        """
        for key, value in other.properties.items():
            if key not in self.properties:
                self.properties[key] = value
            # Could add more sophisticated merging logic here

        # Take higher confidence
        if other.confidence > self.confidence:
            self.confidence = other.confidence

        # Combine grounding refs
        for ref in other.grounding_refs:
            if ref not in self.grounding_refs:
                self.grounding_refs.append(ref)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "embedding": self.embedding,
            "embedding_model": self.embedding_model,
            "confidence": self.confidence,
            "source": self.source.to_dict() if self.source else None,
            "temporal_grain": self.temporal_grain.value,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "grounding_type": self.grounding_type.value,
            "grounding_refs": self.grounding_refs,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            name=data["name"],
            embedding=data.get("embedding"),
            embedding_model=data.get("embedding_model"),
            confidence=data.get("confidence", 0.5),
            source=Source.from_dict(data["source"]) if data.get("source") else None,
            temporal_grain=TemporalGrain(data.get("temporal_grain", "eternal")),
            valid_from=datetime.fromisoformat(data["valid_from"]) if data.get("valid_from") else None,
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None,
            grounding_type=GroundingType(data.get("grounding_type", "ungrounded")),
            grounding_refs=data.get("grounding_refs", []),
            properties=data.get("properties", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else datetime.now(timezone.utc),
            access_count=data.get("access_count", 0),
        )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False


# Factory functions for common node types

def create_concept(
    name: str,
    properties: dict = None,
    source: Source = None,
    confidence: float = 0.7,
) -> Node:
    """Create a node representing an abstract concept."""
    return Node(
        name=name,
        properties=properties or {},
        source=source,
        confidence=confidence,
        temporal_grain=TemporalGrain.ETERNAL,
        grounding_type=GroundingType.UNGROUNDED,
    )


def create_entity(
    name: str,
    properties: dict = None,
    source: Source = None,
    confidence: float = 0.8,
    grounding_refs: list[str] = None,
) -> Node:
    """Create a node representing a concrete entity."""
    return Node(
        name=name,
        properties=properties or {},
        source=source,
        confidence=confidence,
        temporal_grain=TemporalGrain.DAYS,  # Entities typically persist for a while
        grounding_type=GroundingType.BOTH,
        grounding_refs=grounding_refs or [],
    )


def create_event(
    name: str,
    valid_from: datetime,
    valid_until: datetime = None,
    temporal_grain: TemporalGrain = TemporalGrain.SECONDS,
    properties: dict = None,
    source: Source = None,
    confidence: float = 0.8,
) -> Node:
    """Create a node representing an event."""
    return Node(
        name=name,
        properties=properties or {},
        source=source,
        confidence=confidence,
        temporal_grain=temporal_grain,
        valid_from=valid_from,
        valid_until=valid_until,
        grounding_type=GroundingType.PERCEPTUAL,
    )


def create_self() -> Node:
    """Create the special Self node - the agent's representation of itself."""
    return Node(
        id="self",
        name="Self",
        properties={
            "is_agent": True,
            "can_learn": True,
            "can_reason": True,
            "is_uncertain": True,  # Starts uncertain about most things
        },
        source=Source.bootstrap("The agent's self-representation"),
        confidence=1.0,  # We're certain we exist
        temporal_grain=TemporalGrain.ETERNAL,
        grounding_type=GroundingType.BOTH,  # We can perceive and act
    )
