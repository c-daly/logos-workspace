"""
Universal Knowledge Substrate

A minimal, self-extensible knowledge representation that can handle
anything from calculus to falling objects, from instant events to
geological timescales.

The core insight: everything is a node with properties and relations.
Types emerge from patterns, not from pre-assigned categories.
"""

from .node import Node, TemporalGrain, GroundingType
from .edge import Edge
from .source import Source, SourceType
from .graph import KnowledgeGraph

__all__ = [
    "Node",
    "Edge",
    "TemporalGrain",
    "GroundingType",
    "Source",
    "SourceType",
    "KnowledgeGraph",
]
