"""
Source tracking for epistemic provenance.

Every piece of knowledge needs to know where it came from,
so the system can reason about reliability and update confidence
when sources prove trustworthy or not.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid


class SourceType(Enum):
    """How did this knowledge enter the system?"""

    # Direct experience
    DIRECT_OBSERVATION = "direct_observation"  # Agent perceived it
    SENSOR_READING = "sensor_reading"          # Hardware measurement
    ACTION_OUTCOME = "action_outcome"          # Result of agent's action

    # External input
    USER_STATEMENT = "user_statement"          # Human told us
    USER_CORRECTION = "user_correction"        # Human corrected us (high signal)
    DOCUMENT = "document"                      # Read from text

    # Generated
    LLM_EXTRACTION = "llm_extraction"          # LLM parsed from conversation
    LLM_INFERENCE = "llm_inference"            # LLM reasoned to conclusion

    # Internal
    INFERENCE = "inference"                    # Derived from other knowledge
    MEMORY_RECALL = "memory_recall"            # Retrieved from storage
    BOOTSTRAP = "bootstrap"                    # Initial seed knowledge

    # Meta
    SELF_REFLECTION = "self_reflection"        # Agent reasoning about itself
    ONTOLOGY_MODIFICATION = "ontology_modification"  # System modified itself


# Base reliability priors - can be learned and adjusted over time
SOURCE_RELIABILITY_PRIORS = {
    SourceType.DIRECT_OBSERVATION: 0.90,
    SourceType.SENSOR_READING: 0.85,
    SourceType.ACTION_OUTCOME: 0.88,
    SourceType.USER_STATEMENT: 0.70,
    SourceType.USER_CORRECTION: 0.85,  # Corrections are usually right
    SourceType.DOCUMENT: 0.65,
    SourceType.LLM_EXTRACTION: 0.60,
    SourceType.LLM_INFERENCE: 0.50,    # LLMs hallucinate
    SourceType.INFERENCE: 0.55,         # Depends on inference chain
    SourceType.MEMORY_RECALL: 0.80,
    SourceType.BOOTSTRAP: 0.95,         # Seed knowledge is trusted
    SourceType.SELF_REFLECTION: 0.60,
    SourceType.ONTOLOGY_MODIFICATION: 0.70,
}


@dataclass
class Source:
    """
    Tracks the provenance of a piece of knowledge.

    Every node and edge has a source, so we always know
    where information came from and can update our trust.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_type: SourceType = SourceType.LLM_EXTRACTION

    # When did we learn this?
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # What specifically? (conversation turn, sensor id, document reference)
    reference: Optional[str] = None

    # Raw content that led to this knowledge
    raw_content: Optional[str] = None

    # If derived from other knowledge, what were the inputs?
    derived_from: list[str] = field(default_factory=list)  # Source IDs

    # Reliability (starts at prior, updated over time)
    # Use None as sentinel to distinguish "not provided" from "explicitly set to 0.5"
    reliability: Optional[float] = field(default=None)

    # Track record: how often has this source been right?
    confirmation_count: int = 0
    contradiction_count: int = 0

    def __post_init__(self):
        """Set initial reliability from prior if not specified."""
        if self.reliability is None:
            self.reliability = SOURCE_RELIABILITY_PRIORS.get(
                self.source_type, 0.5
            )

    def record_confirmation(self):
        """This source was confirmed correct."""
        self.confirmation_count += 1
        self._update_reliability()

    def record_contradiction(self):
        """This source was found to be wrong."""
        self.contradiction_count += 1
        self._update_reliability()

    def _update_reliability(self):
        """Bayesian-ish update of reliability based on track record."""
        total = self.confirmation_count + self.contradiction_count
        if total == 0:
            return

        # Mix prior with observed rate
        prior = SOURCE_RELIABILITY_PRIORS.get(self.source_type, 0.5)
        observed_rate = self.confirmation_count / total

        # Weight prior more when we have few observations
        prior_weight = 5 / (5 + total)  # Approaches 0 as total grows

        self.reliability = prior_weight * prior + (1 - prior_weight) * observed_rate

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "id": self.id,
            "source_type": self.source_type.value,
            "timestamp": self.timestamp.isoformat(),
            "reference": self.reference,
            "raw_content": self.raw_content,
            "derived_from": self.derived_from,
            "reliability": self.reliability,
            "confirmation_count": self.confirmation_count,
            "contradiction_count": self.contradiction_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Source":
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            source_type=SourceType(data["source_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            reference=data.get("reference"),
            raw_content=data.get("raw_content"),
            derived_from=data.get("derived_from", []),
            reliability=data.get("reliability", 0.5),
            confirmation_count=data.get("confirmation_count", 0),
            contradiction_count=data.get("contradiction_count", 0),
        )

    @classmethod
    def bootstrap(cls, description: str = "Initial seed knowledge") -> "Source":
        """Create a source for bootstrap/seed knowledge."""
        return cls(
            source_type=SourceType.BOOTSTRAP,
            reference="bootstrap",
            raw_content=description,
            reliability=0.95,
        )

    @classmethod
    def from_user(cls, content: str, is_correction: bool = False) -> "Source":
        """Create a source from user input."""
        return cls(
            source_type=SourceType.USER_CORRECTION if is_correction else SourceType.USER_STATEMENT,
            raw_content=content,
        )

    @classmethod
    def from_llm(cls, content: str, is_inference: bool = False) -> "Source":
        """Create a source from LLM processing."""
        return cls(
            source_type=SourceType.LLM_INFERENCE if is_inference else SourceType.LLM_EXTRACTION,
            raw_content=content,
        )
