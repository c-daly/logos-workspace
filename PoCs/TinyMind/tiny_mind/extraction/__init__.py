"""
Knowledge Extraction

Uses an LLM to extract structured knowledge from natural language.
The extracted propositions are then integrated into the knowledge graph.
"""

from .extractor import Extractor, ExtractionResult
from .prompts import EXTRACTION_SYSTEM_PROMPT

__all__ = [
    "Extractor",
    "ExtractionResult",
    "EXTRACTION_SYSTEM_PROMPT",
]
