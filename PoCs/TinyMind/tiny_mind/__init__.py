"""
Tiny Mind - A baby intelligence that learns through conversation.

This is a proof-of-concept for a minimal, self-extending knowledge
representation that can handle anything from calculus to falling objects.

Usage:
    from tiny_mind import TinyMind

    mind = TinyMind(name="Tiny")
    print(mind.greet())

    response = mind.hear("The derivative of x squared is 2x")
    print(response)

    response = mind.hear("Objects fall due to gravity")
    print(response)

    print(mind.know())  # See what it learned

Or run interactively:
    python -m tiny_mind.conversation.mind
"""

from .substrate import Node, Edge, KnowledgeGraph, Source, SourceType
from .conversation import TinyMind

__all__ = [
    "TinyMind",
    "Node",
    "Edge",
    "KnowledgeGraph",
    "Source",
    "SourceType",
]

__version__ = "0.1.0"
