"""
Curiosity goal types and data structures.

Goals represent things TinyMind is curious about and wants to investigate.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class GoalType(Enum):
    """Types of curiosity goals."""
    
    # "What don't I know?" - Missing definitions, unexplained concepts
    GAP = "gap"
    
    # "How do things relate?" - Disconnected concepts that might connect
    CONNECTION = "connection"
    
    # "What am I unsure about?" - Low confidence, single source
    UNCERTAINTY = "uncertainty"
    
    # "Can I learn more?" - Important but shallow concepts
    DEPTH = "depth"
    
    # "What's new or adjacent?" - Related concepts not yet explored
    NOVELTY = "novelty"
    
    # "Is this actually true?" - Structural anomalies, suspicious claims
    VERIFICATION = "verification"


# Base priority weights for each goal type
GOAL_TYPE_WEIGHTS = {
    GoalType.GAP: 0.9,           # Missing knowledge is high priority
    GoalType.VERIFICATION: 0.85,  # Wrong info is dangerous
    GoalType.UNCERTAINTY: 0.7,    # Reduce uncertainty
    GoalType.CONNECTION: 0.6,     # Connect knowledge
    GoalType.DEPTH: 0.5,          # Deepen understanding
    GoalType.NOVELTY: 0.4,        # Explore new areas
}


@dataclass
class CuriosityGoal:
    """A goal that TinyMind wants to investigate."""
    
    type: GoalType
    target: str                    # Node name or concept
    question: str                  # Natural language question to investigate
    priority: float                # 0-1, higher = more important
    context: dict = field(default_factory=dict)  # Additional context for investigation
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Optional: specific nodes/edges involved
    related_nodes: list[str] = field(default_factory=list)
    related_edges: list[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"[{self.type.value.upper()}] {self.question} (priority: {self.priority:.2f})"
    
    def to_search_query(self) -> str:
        """Convert goal to a web search query."""
        if self.type == GoalType.GAP:
            return f"definition of {self.target}"
        elif self.type == GoalType.VERIFICATION:
            return f"is it true that {self.question.lower().replace('?', '')}"
        elif self.type == GoalType.CONNECTION:
            # Extract the two concepts from context
            concepts = self.context.get("concepts", [self.target])
            if len(concepts) >= 2:
                return f"relationship between {concepts[0]} and {concepts[1]}"
            return f"what is {self.target} related to"
        elif self.type == GoalType.DEPTH:
            return f"{self.target} explained in detail"
        elif self.type == GoalType.NOVELTY:
            return f"concepts related to {self.target}"
        else:  # UNCERTAINTY
            return f"what is {self.target}"


@dataclass
class InvestigationResult:
    """Result of investigating a curiosity goal."""
    
    goal: CuriosityGoal
    success: bool
    
    # What was learned
    nodes_added: int = 0
    edges_added: int = 0
    nodes_updated: int = 0
    
    # Sources consulted
    sources: list[str] = field(default_factory=list)
    
    # Summary of findings
    summary: str = ""
    
    # If verification, was the claim confirmed?
    verified: Optional[bool] = None
    
    # Any errors encountered
    errors: list[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.success:
            parts = [f"Investigated: {self.goal.question}"]
            if self.nodes_added or self.edges_added:
                parts.append(f"  Learned: +{self.nodes_added} nodes, +{self.edges_added} edges")
            if self.nodes_updated:
                parts.append(f"  Updated: {self.nodes_updated} nodes")
            if self.verified is not None:
                parts.append(f"  Verified: {self.verified}")
            if self.summary:
                parts.append(f"  Summary: {self.summary[:100]}...")
            return "\n".join(parts)
        else:
            return f"Failed to investigate: {self.goal.question}\n  Errors: {self.errors}"
