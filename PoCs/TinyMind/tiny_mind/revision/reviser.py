"""
Knowledge revision system for TinyMind.

Performs maintenance on the knowledge graph:
- Deduplication: Merge semantically similar nodes
- Corroboration: Increase confidence for repeated facts
- Contradiction resolution: Handle conflicting relations
- Pruning: Remove low-confidence orphan nodes
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional

from tiny_mind.substrate.graph import KnowledgeGraph
from tiny_mind.substrate.node import Node
from tiny_mind.substrate.edge import Edge


# Opposite relation pairs for contradiction detection
OPPOSITE_RELATIONS = {
    "causes": "prevents",
    "prevents": "causes",
    "enables": "disables",
    "disables": "enables",
    "supports": "contradicts",
    "contradicts": "supports",
    "increases": "decreases",
    "decreases": "increases",
    "requires": "excludes",
    "excludes": "requires",
    "is_a": "is_not_a",
    "is_not_a": "is_a",
}


@dataclass
class RevisionResult:
    """Result of a revision pass."""
    merged_nodes: list[tuple[str, str]] = field(default_factory=list)  # (kept, removed)
    undone_merges: list[tuple[str, str]] = field(default_factory=list)  # (was_kept, recreated)
    ephemeral_removed: list[str] = field(default_factory=list)  # removed ephemeral/boilerplate nodes
    corroborated: list[str] = field(default_factory=list)  # nodes with increased confidence
    contradictions_resolved: list[dict] = field(default_factory=list)  # resolved conflicts
    pruned: list[str] = field(default_factory=list)  # removed nodes
    refined: list[str] = field(default_factory=list)  # nodes with updated properties
    inferred_relations: list[dict] = field(default_factory=list)  # new hierarchical relations
    split_nodes: list[dict] = field(default_factory=list)  # polysemous nodes split into domain-specific variants
    orphans_rehomed: list[tuple[str, str]] = field(default_factory=list)  # (orphan, merged_into)
    orphans_related: int = 0  # Count of orphans connected via related_to
    orphans_linked_to_base: int = 0  # Count of orphans linked to Thing

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        if self.undone_merges:
            lines.append(f"Undid {len(self.undone_merges)} bad merges")
        if self.ephemeral_removed:
            lines.append(f"Removed {len(self.ephemeral_removed)} ephemeral/boilerplate nodes")
        if self.merged_nodes:
            lines.append(f"Merged {len(self.merged_nodes)} duplicate node pairs")
        if self.inferred_relations:
            lines.append(f"Inferred {len(self.inferred_relations)} hierarchical relations")
        if self.split_nodes:
            lines.append(f"Split {len(self.split_nodes)} polysemous nodes")
        if self.orphans_rehomed:
            lines.append(f"Rehomed {len(self.orphans_rehomed)} orphan nodes (merged)")
        if self.orphans_related:
            lines.append(f"Connected {self.orphans_related} orphans via related_to")
        if self.orphans_linked_to_base:
            lines.append(f"Linked {self.orphans_linked_to_base} orphans to base concept")
        if self.corroborated:
            lines.append(f"Corroborated {len(self.corroborated)} nodes")
        if self.contradictions_resolved:
            lines.append(f"Resolved {len(self.contradictions_resolved)} contradictions")
        if self.pruned:
            lines.append(f"Pruned {len(self.pruned)} low-value nodes")
        if self.refined:
            lines.append(f"Refined {len(self.refined)} nodes")
        return "\n".join(lines) if lines else "No changes made"


class RevisionLog:
    """Persistent log of all revision actions."""

    def __init__(self, log_path: str = "revision_log.json"):
        self.log_path = log_path
        self.entries: list[dict] = []
        self._load()

    def _load(self):
        """Load existing log if present."""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    self.entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.entries = []

    def _save(self):
        """Persist log to disk."""
        with open(self.log_path, "w") as f:
            json.dump(self.entries, f, indent=2, default=str)

    def log(self, action_type: str, details: dict):
        """Log a revision action."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": action_type,
            **details,
        }
        self.entries.append(entry)
        self._save()

    def log_merge(self, kept: str, removed: str, similarity: float, reason: str, 
                  redirected_edges: list[dict] = None):
        """Log a node merge with optional edge information for undo support."""
        entry = {
            "kept": kept,
            "removed": removed,
            "similarity": similarity,
            "reason": reason,
        }
        if redirected_edges:
            entry["redirected_edges"] = redirected_edges
        self.log("merge", entry)

    def log_action(self, entry: dict):
        """Log an arbitrary action."""
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.entries.append(entry)
        self._save()

    def read_log(self) -> list[dict]:
        """Return all log entries."""
        return self.entries

    def log_contradiction(self, kept_edge: dict, removed_edge: dict, reason: str):
        """Log a contradiction resolution."""
        self.log("contradiction_resolved", {
            "kept": kept_edge,
            "removed": removed_edge,
            "reason": reason,
        })

    def log_prune(self, node_name: str, confidence: float, reason: str):
        """Log a node pruning."""
        self.log("prune", {
            "node": node_name,
            "confidence": confidence,
            "reason": reason,
        })

    def log_corroboration(self, node_name: str, old_confidence: float, new_confidence: float):
        """Log a confidence increase."""
        self.log("corroboration", {
            "node": node_name,
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
        })


class Reviser:
    """
    Knowledge graph maintenance system.
    
    Performs deduplication, corroboration, contradiction resolution, and pruning.
    All actions are auto-accepted and logged.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = "gpt-4o",
        similarity_threshold: float = 0.92,
        embedding_model: str = "text-embedding-3-small",
        log_path: str = "revision_log.json",
        embedding_cache_path: str = "embedding_cache.json",
        prune_confidence_threshold: float = 0.3,
        prune_staleness_days: int = 7,
    ):
        self.llm_provider = llm_provider
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.prune_confidence_threshold = prune_confidence_threshold
        self.prune_staleness_days = prune_staleness_days
        self.embedding_cache_path = embedding_cache_path
        
        self.log = RevisionLog(log_path)
        self._embedding_client = None
        self._embedding_cache: dict[str, list[float]] = {}
        self._cache_dirty = False
        
        # Learned domain relationships (populated by _learn_domain_relationships)
        self._learned_domain_compatibility: dict[tuple[str, str], float] = {}
        
        # Load persistent cache
        self._load_embedding_cache()

    def _load_embedding_cache(self):
        """Load embedding cache from disk."""
        import json
        from pathlib import Path
        
        cache_path = Path(self.embedding_cache_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    self._embedding_cache = data.get("embeddings", {})
            except (json.JSONDecodeError, IOError):
                self._embedding_cache = {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        import json
        from pathlib import Path
        
        if not self._cache_dirty:
            return
        
        cache_path = Path(self.embedding_cache_path)
        try:
            with open(cache_path, 'w') as f:
                json.dump({"embeddings": self._embedding_cache}, f)
            self._cache_dirty = False
        except IOError as e:
            print(f"Warning: Could not save embedding cache: {e}")

    def _get_embedding_client(self):
        """Get or create embedding client."""
        if self._embedding_client is None:
            if self.llm_provider == "openai":
                from openai import OpenAI
                self._embedding_client = OpenAI()
            else:
                raise ValueError(f"Embeddings not supported for provider: {self.llm_provider}")
        return self._embedding_client

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text, with persistent caching."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        client = self._get_embedding_client()
        response = client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        embedding = response.data[0].embedding
        self._embedding_cache[text] = embedding
        self._cache_dirty = True
        return embedding

    def _get_embeddings_batch(self, texts: list[str], batch_size: int = 100) -> dict[str, list[float]]:
        """
        Get embeddings for multiple texts efficiently using batch API calls.
        
        Returns dict mapping text -> embedding.
        """
        result = {}
        uncached = []
        
        # Check cache first
        for text in texts:
            if text in self._embedding_cache:
                result[text] = self._embedding_cache[text]
            else:
                uncached.append(text)
        
        if not uncached:
            return result
        
        # Batch API calls
        client = self._get_embedding_client()
        
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            try:
                response = client.embeddings.create(
                    model=self.embedding_model,
                    input=batch,
                )
                for j, emb_data in enumerate(response.data):
                    text = batch[j]
                    embedding = emb_data.embedding
                    self._embedding_cache[text] = embedding
                    result[text] = embedding
                    self._cache_dirty = True
            except Exception as e:
                # Fall back to individual calls
                for text in batch:
                    try:
                        result[text] = self._get_embedding(text)
                    except Exception:
                        pass
        
        return result
    
    def _precompute_embeddings(self, nodes: list, verbose: bool = False) -> dict[str, list[float]]:
        """
        Precompute embeddings for all nodes using batch API calls.
        
        Returns dict mapping node_id -> embedding.
        """
        # Build text representation for each node
        def node_text(n) -> str:
            parts = [n.name]
            if n.properties.get("definition"):
                parts.append(n.properties["definition"])
            if n.properties.get("is_a"):
                is_a = n.properties["is_a"]
                if isinstance(is_a, list):
                    is_a = is_a[0] if is_a else ""
                parts.append(f"is a {is_a}")
            return " ".join(parts)
        
        # Collect texts that need embeddings
        texts_to_embed = []
        node_to_text = {}
        
        for node in nodes:
            text = node_text(node)
            node_to_text[node.id] = text
            if text not in self._embedding_cache:
                texts_to_embed.append(text)
        
        if verbose and texts_to_embed:
            print(f"  Precomputing {len(texts_to_embed)} embeddings ({len(nodes) - len(texts_to_embed)} cached)...")
        
        # Batch compute uncached embeddings
        if texts_to_embed:
            self._get_embeddings_batch(texts_to_embed)
        
        # Build result mapping
        result = {}
        for node_id, text in node_to_text.items():
            if text in self._embedding_cache:
                result[node_id] = self._embedding_cache[text]
        
        return result

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _string_similarity(self, a: str, b: str) -> float:
        """Compute string similarity using SequenceMatcher."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _is_plural_pair(self, a: str, b: str) -> bool:
        """Check if one string is plural of the other."""
        a_lower, b_lower = a.lower(), b.lower()
        # Simple English pluralization rules
        if a_lower + "s" == b_lower or b_lower + "s" == a_lower:
            return True
        if a_lower + "es" == b_lower or b_lower + "es" == a_lower:
            return True
        # cities/city (endswith, not rstrip!)
        if a_lower.endswith("ies") and a_lower[:-3] + "y" == b_lower:
            return True
        if b_lower.endswith("ies") and b_lower[:-3] + "y" == a_lower:
            return True
        # matrices/matrix
        if a_lower.endswith("ices") and a_lower[:-4] + "ix" == b_lower:
            return True
        if b_lower.endswith("ices") and b_lower[:-4] + "ix" == a_lower:
            return True
        return False

    # Contrasting/opposite word pairs that should never be merged
    # Contrasting/opposite word pairs that should never be merged
    OPPOSITE_WORDS = [
        ("odd", "even"), ("left", "right"), ("up", "down"),
        ("increasing", "decreasing"), ("positive", "negative"),
        ("horizontal", "vertical"), ("x", "y"), ("sin", "cos"),
        ("sine", "cosine"), ("linear", "nonlinear"), ("open", "closed"),
        ("maximum", "minimum"), ("max", "min"), ("upper", "lower"),
        ("before", "after"), ("input", "output"), ("domain", "range"),
        ("source", "target"), ("start", "end"), ("first", "last"),
        ("natural", "common"), ("inverse", "direct"), ("arrival", "departure"),
        ("in", "out"), ("internal", "external"), ("local", "global"),
        ("interval", "integral"), ("tangent", "secant"), ("real", "complex"),
        ("rational", "irrational"), ("finite", "infinite"), ("continuous", "discrete"),
        ("celsius", "fahrenheit"), ("square", "cube"), ("addition", "subtraction"),
        ("sum", "difference"), ("product", "quotient"), ("numerator", "denominator"),
        # Matrix/linear algebra distinctions
        ("regular", "singular"), ("regular", "real"), ("real", "imaginary"),
        ("symmetric", "antisymmetric"), ("symmetric", "skew"), ("sparse", "dense"),
        ("diagonal", "triangular"), ("orthogonal", "orthonormal"),
        ("row", "column"), ("invertible", "singular"),
        # Biology distinctions
        ("vertebrate", "invertebrate"), ("prokaryote", "eukaryote"),
        ("plant", "animal"), ("predator", "prey"), ("male", "female"),
        ("mitosis", "meiosis"), ("dna", "rna"), ("aerobic", "anaerobic"),
        # Physics/chemistry distinctions  
        ("proton", "neutron"), ("proton", "electron"), ("neutron", "electron"),
        ("kinetic", "potential"), ("static", "dynamic"), ("solid", "liquid"),
        ("endothermic", "exothermic"), ("acid", "base"), ("cation", "anion"),
        # Action distinctions
        ("carry", "strike"), ("push", "pull"), ("send", "receive"),
        ("read", "write"), ("load", "save"), ("import", "export"),
    ]

    # Domain groups that are related/compatible - nodes with qualifiers from the same group can merge
    # Each tuple is a group of domains that represent the same "area of knowledge"
    RELATED_DOMAIN_GROUPS = [
        # STEM / Mathematics family - these are all compatible
        {"mathematics", "math", "applied mathematics", "pure mathematics", "discrete mathematics",
         "linear algebra", "calculus", "algebra", "geometry", "trigonometry", "statistics",
         "probability", "optimization", "numerical methods", "analysis"},
        # Computer Science / ML family  
        {"computer science", "cs", "machine learning", "ml", "deep learning", "ai",
         "artificial intelligence", "data science", "programming", "software"},
        # Physics family
        {"physics", "mechanics", "thermodynamics", "electromagnetism", "quantum mechanics",
         "classical mechanics", "optics"},
        # Chemistry family
        {"chemistry", "organic chemistry", "inorganic chemistry", "biochemistry"},
        # Biology family
        {"biology", "genetics", "ecology", "zoology", "botany", "microbiology",
         "paleontology", "evolution"},
        # Geography / Earth sciences
        {"geography", "geology", "meteorology", "oceanography", "cartography"},
        # Psychology / Cognitive sciences
        {"psychology", "cognitive science", "neuroscience", "psychiatry", "behavioral science"},
        # Business / Economics
        {"business", "economics", "finance", "accounting", "marketing", "management",
         "corporate structure", "telecommunications"},
        # History / Humanities
        {"history", "philosophy", "literature", "linguistics", "anthropology"},
    ]
    
    # "Technical" domains and generic qualifiers - when one node is unqualified and 
    # the other has one of these, it's usually just metadata, not disambiguation (so allow merge)
    METADATA_DOMAINS = {
        # STEM fields
        "mathematics", "math", "applied mathematics", "pure mathematics",
        "machine learning", "ml", "deep learning", "ai", "artificial intelligence",
        "computer science", "cs", "programming", "software",
        "optimization", "statistics", "probability", "data science",
        "physics", "mechanics", "quantum mechanics",
        "trigonometry", "geometry", "linear algebra", "calculus", "algebra", 
        "analysis", "topology", "number theory",
        "science", "engineering",
        # Generic/non-domain qualifiers (just metadata, not disambiguation)
        "general", "basic", "advanced", "introduction", "intro", "overview",
        "definition", "concept", "theory", "application", "applications",
        "example", "examples", "properties", "fundamentals",
    }

    # Words that look similar but have completely different meanings
    CONFUSABLE_WORDS = [
        ("neutron", "neuron"), ("neural", "neutral"),
        ("causal", "casual"), ("affect", "effect"),
        ("complement", "compliment"), ("principal", "principle"),
        ("stationary", "stationery"), ("discrete", "discreet"),
        ("emigrate", "immigrate"), ("elicit", "illicit"),
        ("induction", "deduction"), ("function", "junction"),
        ("adaption", "adoption"), ("allusion", "illusion"),
        ("moral", "morale"), ("personal", "personnel"),
    ]

    def _should_skip_merge(self, name1: str, name2: str) -> bool:
        """Check if these nodes should NOT be merged even if similar."""
        import re

        n1_lower, n2_lower = name1.lower(), name2.lower()

        # Don't merge pure numbers (years, quantities)
        if name1.isdigit() or name2.isdigit():
            return True

        # Don't merge very short names (high false positive risk)
        if len(name1) <= 2 or len(name2) <= 2:
            return True

        # === PROTECT PARENTHETICAL DOMAIN QUALIFIERS (with domain awareness) ===
        # Pattern: "concept (domain)" - the parenthetical may disambiguate polysemy
        qualifier_pattern = r'^(.+?)\s*\(([^)]+)\)$'
        match1 = re.match(qualifier_pattern, name1)
        match2 = re.match(qualifier_pattern, name2)
        
        def get_domain_group(qualifier: str) -> set | None:
            """Find which domain group a qualifier belongs to."""
            qual_lower = qualifier.lower().strip()
            for group in self.RELATED_DOMAIN_GROUPS:
                if qual_lower in group:
                    return group
            return None
        
        def domains_are_compatible(qual1: str, qual2: str) -> bool:
            """Check if two qualifiers are from compatible/related domains.
            
            Uses learned relationships first, falls back to hardcoded groups.
            """
            q1_lower = qual1.lower().strip()
            q2_lower = qual2.lower().strip()
            
            # Exact match
            if q1_lower == q2_lower:
                return True
            
            # === FIRST: Check learned domain relationships ===
            learned = self._domains_are_compatible_learned(q1_lower, q2_lower)
            if learned is not None:
                return learned
            
            # === FALLBACK: Use hardcoded domain groups ===
            group1 = get_domain_group(qual1)
            group2 = get_domain_group(qual2)
            
            # STEM groups (indices 0, 1, 2) are all mutually compatible
            # since they share vocabulary with consistent meanings
            stem_groups = {id(self.RELATED_DOMAIN_GROUPS[0]), 
                          id(self.RELATED_DOMAIN_GROUPS[1]), 
                          id(self.RELATED_DOMAIN_GROUPS[2])}
            
            if group1 and group2:
                # Same group = compatible
                if group1 is group2:
                    return True
                # Both STEM = compatible (math, CS, physics share concepts)
                if id(group1) in stem_groups and id(group2) in stem_groups:
                    return True
                return False
            
            # One or both unknown - use string similarity as fallback
            # High similarity suggests related domains
            return self._string_similarity(q1_lower, q2_lower) > 0.7
        
        if match1 or match2:
            # Extract base names and qualifiers
            base1 = match1.group(1).strip().lower() if match1 else n1_lower
            base2 = match2.group(1).strip().lower() if match2 else n2_lower
            qual1 = match1.group(2).strip().lower() if match1 else None
            qual2 = match2.group(2).strip().lower() if match2 else None
            
            # Check if bases are similar
            bases_similar = (
                base1 == base2 or
                base1 in base2 or 
                base2 in base1 or
                self._string_similarity(base1, base2) > 0.8
            )
            
            if bases_similar:
                # Case 1: Both have qualifiers
                if qual1 and qual2:
                    # If base names are IDENTICAL and qualifiers are DIFFERENT,
                    # they were explicitly disambiguated - never merge
                    if base1 == base2 and qual1 != qual2:
                        return True  # Explicit disambiguation - don't merge

                    # For similar-but-not-identical bases, check domain compatibility
                    if not domains_are_compatible(qual1, qual2):
                        return True
                
                # Case 2: One has qualifier, one doesn't
                elif qual1 or qual2:
                    qualifier = qual1 or qual2
                    base_with_qual = base1 if qual1 else base2
                    name_without_qual = n2_lower if qual1 else n1_lower

                    # Check if unqualified name matches the base
                    if (base_with_qual == name_without_qual or
                        self._string_similarity(base_with_qual, name_without_qual) > 0.9):

                        # FIRST: If qualifier is metadata/STEM domain, always allow merge
                        # These are just annotations, not semantic disambiguation
                        if qualifier in self.METADATA_DOMAINS:
                            pass  # Allow merge - it's just metadata
                        else:
                            # For non-metadata qualifiers, check if it's domain disambiguation
                            # Only block if it's a non-STEM domain
                            group = get_domain_group(qualifier)
                            if group and group not in [self.RELATED_DOMAIN_GROUPS[0],
                                                        self.RELATED_DOMAIN_GROUPS[1],
                                                        self.RELATED_DOMAIN_GROUPS[2]]:
                                # Non-STEM domain qualifier - might be disambiguation
                                return True

        # Don't merge if names contain opposite words
        for w1, w2 in self.OPPOSITE_WORDS:
            if (w1 in n1_lower and w2 in n2_lower) or (w2 in n1_lower and w1 in n2_lower):
                return True

        # Don't merge confusable word pairs
        for w1, w2 in self.CONFUSABLE_WORDS:
            # Check if one name contains w1 and other contains w2
            if ((w1 in n1_lower and w2 in n2_lower) or 
                (w2 in n1_lower and w1 in n2_lower)):
                return True
            # Also check if the names ARE these words (with possible suffixes)
            if ((n1_lower.startswith(w1) and n2_lower.startswith(w2)) or
                (n1_lower.startswith(w2) and n2_lower.startswith(w1))):
                return True

        # Don't merge numbered items (examples, exercises, chapters, etc.)
        # More aggressive pattern - any item with a number
        numbered_pattern = r'^(example|section|chapter|figure|table|equation|theorem|lemma|definition|exercise|rule|step|part|item|case|type|version|phase|stage|level|round|day|week|month|year)\s*[\d\.\-:]'
        if (re.match(numbered_pattern, n1_lower) and
            re.match(numbered_pattern, n2_lower)):
            # Only skip if the numbers are different
            num1 = re.findall(r'[\d]+(?:\.[\d]+)*', name1)
            num2 = re.findall(r'[\d]+(?:\.[\d]+)*', name2)
            if num1 != num2:
                return True

        # Don't merge "X N.N.N" patterns (like "Exercise 2.7.11" vs "Exercise 1.2.25")
        versioned_pattern = r'.*\d+\.\d+'
        if re.match(versioned_pattern, n1_lower) and re.match(versioned_pattern, n2_lower):
            nums1 = re.findall(r'\d+\.\d+(?:\.\d+)*', name1)
            nums2 = re.findall(r'\d+\.\d+(?:\.\d+)*', name2)
            if nums1 and nums2 and nums1 != nums2:
                return True

        # Don't merge matrix A with matrix B (explicit variable names)
        var_pattern = r'^(matrix|vector|scalar|variable|function|set|point|line|plane)\s+[A-Za-z]$'
        if (re.match(var_pattern, n1_lower) or
            re.match(var_pattern, n2_lower)):
            return True

        # Don't merge different function types (sine vs linear, etc.)
        func_types = ['sine', 'cosine', 'linear', 'quadratic', 'cubic', 'exponential',
                      'logarithmic', 'polynomial', 'rational', 'trigonometric', 'power', 'root']
        n1_types = [t for t in func_types if t in n1_lower]
        n2_types = [t for t in func_types if t in n2_lower]
        if n1_types and n2_types and n1_types != n2_types:
            return True

        # Don't merge if both start with "rule:" but have different key verbs
        if n1_lower.startswith("rule:") and n2_lower.startswith("rule:"):
            key_verbs = ["carry", "strike", "hit", "throw", "catch", "run", "walk", 
                         "is", "are", "has", "have", "can", "must", "should"]
            n1_verbs = [v for v in key_verbs if v in n1_lower]
            n2_verbs = [v for v in key_verbs if v in n2_lower]
            if n1_verbs and n2_verbs and set(n1_verbs) != set(n2_verbs):
                return True

        return False

    def _might_be_duplicate(self, n1: Node, n2: Node) -> tuple[bool, float, str]:
        """
        Quick check if two nodes might be duplicates.
        Returns (is_candidate, similarity, reason).
        """
        name1, name2 = n1.name, n2.name
        
        # Check exclusions first
        if self._should_skip_merge(name1, name2):
            return False, 0.0, ""
        
        # Exact match (shouldn't happen but handle it)
        if name1.lower() == name2.lower():
            return True, 1.0, "exact_match"
        
        # Plural pairs
        if self._is_plural_pair(name1, name2):
            return True, 0.95, "plural_pair"
        
        # One contains the other as significant substring
        if len(name1) > 3 and len(name2) > 3:
            if name1.lower() in name2.lower() or name2.lower() in name1.lower():
                shorter = min(len(name1), len(name2))
                longer = max(len(name1), len(name2))
                if shorter / longer > 0.5:  # Significant overlap
                    return True, 0.8, "substring"
        
        # String similarity for short names
        str_sim = self._string_similarity(name1, name2)
        if str_sim > 0.85:
            return True, str_sim, "string_similarity"
        
        return False, 0.0, ""

    def _extract_domain_from_name(self, name: str) -> tuple[str, str | None]:
        """Extract base name and domain qualifier from a node name.
        
        Returns (base_name, domain) where domain is None if no qualifier.
        Example: "graph (mathematics)" -> ("graph", "mathematics")
        """
        import re
        match = re.match(r'^(.+?)\s*\(([^)]+)\)$', name)
        if match:
            return match.group(1).strip(), match.group(2).strip().lower()
        return name, None

    def _learn_domain_relationships(
        self, 
        graph: "KnowledgeGraph",
        verbose: bool = True,
        min_nodes_per_domain: int = 2,
        compatibility_threshold: float = 0.3,
    ) -> dict[tuple[str, str], float]:
        """Learn domain compatibility from graph structure.
        
        Analyzes how nodes with different domain qualifiers connect to infer
        which domains are related. Stores results in self._learned_domain_compatibility.
        
        Compatibility is computed from:
        1. Shared neighbors - domains whose nodes connect to similar concepts
        2. Direct edges - domains with edges between their nodes
        3. Embedding similarity - domains whose nodes cluster together
        
        Returns dict mapping (domain1, domain2) -> compatibility score [0, 1]
        """
        from collections import defaultdict
        import re
        
        if verbose:
            print("  Learning domain relationships from graph structure...")
        
        # Step 1: Extract all domains and their nodes
        domain_nodes: dict[str, list] = defaultdict(list)
        qualifier_pattern = r'^(.+?)\s*\(([^)]+)\)$'
        
        for node in graph.nodes():
            match = re.match(qualifier_pattern, node.name)
            if match:
                domain = match.group(2).strip().lower()
                domain_nodes[domain].append(node)
        
        # Filter domains with enough nodes
        domains = {d: nodes for d, nodes in domain_nodes.items() 
                   if len(nodes) >= min_nodes_per_domain}
        
        if verbose:
            print(f"    Found {len(domains)} domains with {min_nodes_per_domain}+ nodes")
        
        if len(domains) < 2:
            self._learned_domain_compatibility = {}
            return {}
        
        # Step 2: For each domain, collect neighbor node IDs
        domain_neighbors: dict[str, set] = {}
        for domain, nodes in domains.items():
            neighbors = set()
            for node in nodes:
                # Get all connected nodes
                for edge in graph.find_edges(source_id=node.id):
                    neighbors.add(edge.target_id)
                for edge in graph.find_edges(target_id=node.id):
                    neighbors.add(edge.source_id)
            domain_neighbors[domain] = neighbors
        
        # Step 3: Compute pairwise domain compatibility
        compatibility: dict[tuple[str, str], float] = {}
        domain_list = list(domains.keys())
        
        for i, d1 in enumerate(domain_list):
            for d2 in domain_list[i + 1:]:
                n1 = domain_neighbors.get(d1, set())
                n2 = domain_neighbors.get(d2, set())
                
                # Jaccard similarity of neighbor sets
                if n1 or n2:
                    intersection = len(n1 & n2)
                    union = len(n1 | n2)
                    jaccard = intersection / union if union > 0 else 0
                else:
                    jaccard = 0
                
                # Direct edges between domains
                direct_edges = 0
                nodes_d1 = {n.id for n in domains[d1]}
                nodes_d2 = {n.id for n in domains[d2]}
                
                for edge in graph.edges():
                    if ((edge.source_id in nodes_d1 and edge.target_id in nodes_d2) or
                        (edge.source_id in nodes_d2 and edge.target_id in nodes_d1)):
                        direct_edges += 1
                
                # Normalize direct edges by total possible
                max_edges = len(domains[d1]) * len(domains[d2])
                edge_score = min(direct_edges / max(max_edges * 0.1, 1), 1.0)  # Cap at 10% connectivity
                
                # Combined score (weighted average)
                score = 0.6 * jaccard + 0.4 * edge_score
                
                # Store both directions
                key = tuple(sorted([d1, d2]))
                compatibility[key] = score
        
        # Log discovered relationships
        compatible_pairs = [(k, v) for k, v in compatibility.items() if v >= compatibility_threshold]
        if verbose and compatible_pairs:
            compatible_pairs.sort(key=lambda x: -x[1])
            print(f"    Discovered {len(compatible_pairs)} compatible domain pairs:")
            for (d1, d2), score in compatible_pairs[:5]:  # Show top 5
                print(f"      {d1} <-> {d2}: {score:.2f}")
            if len(compatible_pairs) > 5:
                print(f"      ... and {len(compatible_pairs) - 5} more")
        
        self._learned_domain_compatibility = compatibility
        return compatibility

    def _domains_are_compatible_learned(self, qual1: str, qual2: str) -> bool | None:
        """Check if two domains are compatible based on learned relationships.
        
        Returns:
            True if learned to be compatible (score > 0.4)
            False if learned to be incompatible (score < 0.05)
            None if uncertain or no data (should fall back to hardcoded rules)
        """
        if not hasattr(self, '_learned_domain_compatibility') or not self._learned_domain_compatibility:
            return None
        
        q1 = qual1.lower().strip()
        q2 = qual2.lower().strip()
        
        if q1 == q2:
            return True
        
        key = tuple(sorted([q1, q2]))
        if key in self._learned_domain_compatibility:
            score = self._learned_domain_compatibility[key]
            # High score = definitely compatible
            if score > 0.4:
                return True
            # Very low score = definitely incompatible (no shared structure)
            if score < 0.05:
                return False
            # In between = uncertain, use hardcoded fallback
            return None
        
        return None  # No data, fall back to hardcoded

    def _get_contextualized_embedding(
        self, 
        concept: str, 
        domain_label: str,
    ) -> list[float]:
        """Get embedding for a concept combined with a domain label.
        
        Creates a simple compound term like "calculus physics" or "range geography".
        This captures how the concept is understood in that domain.
        """
        compound = f"{concept} {domain_label}"
        return self._get_embedding(compound)

    def _is_true_polysemy(
        self,
        node_name: str,
        clusters: list,  # List of cluster objects with edge_indices
        connected: list,  # List of (edge, other_node) tuples
        similarity_threshold: float = 0.85,
        verbose: bool = True,
    ) -> bool:
        """Check if proposed clusters represent true polysemy using contextualized embeddings.
        
        For each cluster, builds a contextualized embedding of the concept
        in that cluster's context. If the embeddings are similar, the concept
        means the same thing in both contexts (not polysemous).
        If embeddings are dissimilar, it's true polysemy.
        
        Args:
            node_name: The concept being analyzed
            clusters: List of cluster objects with edge_indices attribute
            connected: List of (edge, other_node) pairs
            similarity_threshold: Above this = same meaning = not polysemous
            verbose: Whether to print debug info
            
        Returns:
            True if this is genuine polysemy (should split)
            False if same concept in different contexts (should NOT split)
        """
        if len(clusters) < 2:
            return False
        
        # Build contextualized embeddings for each cluster
        cluster_embeddings = []
        cluster_contexts = []
        
        for cluster in clusters:
            # Get nodes in this cluster
            context_nodes = []
            for idx in cluster.edge_indices:
                if 0 <= idx < len(connected):
                    _, other_node = connected[idx]
                    context_nodes.append(other_node.name)
            
            if not context_nodes:
                continue
            
            cluster_contexts.append(context_nodes)
            embedding = self._get_contextualized_embedding(node_name, context_nodes)
            cluster_embeddings.append(embedding)
        
        if len(cluster_embeddings) < 2:
            return False
        
        # Compare all pairs of cluster embeddings
        min_similarity = 1.0
        for i in range(len(cluster_embeddings)):
            for j in range(i + 1, len(cluster_embeddings)):
                sim = self._cosine_similarity(cluster_embeddings[i], cluster_embeddings[j])
                min_similarity = min(min_similarity, sim)
                
                if verbose:
                    ctx1_preview = ", ".join(cluster_contexts[i][:3])
                    ctx2_preview = ", ".join(cluster_contexts[j][:3])
                    print(f"      Context similarity: {sim:.3f}")
                    print(f"        [{ctx1_preview}...] vs [{ctx2_preview}...]")
        
        # High similarity = same meaning = NOT polysemous
        # Low similarity = different meanings = IS polysemous
        is_polysemous = min_similarity < similarity_threshold
        
        if verbose:
            if is_polysemous:
                print(f"      → Contexts are dissimilar ({min_similarity:.3f} < {similarity_threshold}) - TRUE polysemy")
            else:
                print(f"      → Contexts are similar ({min_similarity:.3f} >= {similarity_threshold}) - same concept, different applications")
        
        return is_polysemous


    def _get_semantic_dimensions(
        self, 
        word: str,
        verbose: bool = False,
    ) -> tuple[list[str], bool]:
        """Get the distinct semantic dimensions a word can occupy.
        
        Asks the LLM what genuinely different meanings a word can have.
        Results are cached to avoid repeated calls.
        
        Args:
            word: The word to analyze
            verbose: Whether to print debug info
            
        Returns:
            Tuple of (list of dimension labels, is_polysemous)
        """
        # Check cache first
        if not hasattr(self, '_semantic_dimensions_cache'):
            self._semantic_dimensions_cache = {}
        
        cache_key = word.lower().strip()
        if cache_key in self._semantic_dimensions_cache:
            return self._semantic_dimensions_cache[cache_key]
        
        from tiny_mind.extraction.prompts import SEMANTIC_DIMENSIONS_PROMPT
        from tiny_mind.extraction.schemas import SemanticDimensionsSchema
        
        prompt = SEMANTIC_DIMENSIONS_PROMPT.format(word=word)
        
        try:
            if self.llm_provider == "openai":
                from openai import OpenAI
                client = OpenAI()
                
                response = client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You identify genuinely distinct referents of words, not different descriptions of the same thing."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    response_format=SemanticDimensionsSchema,
                )
                result = response.choices[0].message.parsed
                dimensions = result.dimensions
                is_polysemous = result.is_polysemous
                
                if verbose:
                    print(f"    Semantic dimensions of '{word}': {dimensions}")
                    print(f"    Is polysemous: {is_polysemous}")
                
            else:
                if verbose:
                    print(f"    Semantic dimensions only supported with OpenAI provider")
                dimensions = []
                is_polysemous = False
                
        except Exception as e:
            if verbose:
                print(f"    Failed to get semantic dimensions: {e}")
            dimensions = []
            is_polysemous = False
        
        # Cache the result
        self._semantic_dimensions_cache[cache_key] = (dimensions, is_polysemous)
        return dimensions, is_polysemous

    def _compute_semantic_similarity(self, n1: Node, n2: Node) -> float:
        """Compute semantic similarity using embeddings."""
        # Build rich text representation including properties
        def node_text(n: Node) -> str:
            parts = [n.name]
            if n.properties.get("definition"):
                parts.append(n.properties["definition"])
            if n.properties.get("is_a"):
                parts.append(f"is a {n.properties['is_a']}")
            return " ".join(parts)
        
        emb1 = self._get_embedding(node_text(n1))
        emb2 = self._get_embedding(node_text(n2))
        return self._cosine_similarity(emb1, emb2)

    def _deduplicate(self, graph: KnowledgeGraph, verbose: bool = True) -> list[tuple[str, str]]:
        """
        Find and merge duplicate nodes using blocking for efficiency.
        
        Uses prefix blocking to reduce O(n²) to O(n×k) comparisons.
        Precomputes embeddings in batch for efficiency.
        """
        from collections import defaultdict
        
        merged = []
        nodes = list(graph.nodes())
        
        # Skip meta nodes
        skip_names = {"Self", "Thing"}
        nodes = [n for n in nodes if n.name not in skip_names]
        
        if verbose:
            print(f"Checking {len(nodes)} nodes for duplicates (using blocking)...")
        
        # Precompute embeddings for all nodes in batch
        node_embeddings = self._precompute_embeddings(nodes, verbose)
        
        # Build blocks by 3-char prefix
        blocks: dict[str, list] = defaultdict(list)
        for node in nodes:
            name_lower = node.name.lower()
            if len(name_lower) >= 3:
                prefix = name_lower[:3]
                blocks[prefix].append(node)
            else:
                blocks["_short"].append(node)  # Group short names together
        
        # Count candidate pairs for progress
        candidate_pairs = sum(len(block) * (len(block) - 1) // 2 for block in blocks.values())
        if verbose:
            print(f"  {len(blocks)} blocks, {candidate_pairs} candidate pairs (vs {len(nodes)*(len(nodes)-1)//2} full)")
        
        # Track which nodes have been merged away
        merged_away = set()
        
        # Helper to compute similarity using precomputed embeddings
        def fast_semantic_similarity(n1, n2) -> float:
            emb1 = node_embeddings.get(n1.id)
            emb2 = node_embeddings.get(n2.id)
            if emb1 is None or emb2 is None:
                return 0.0
            return self._cosine_similarity(emb1, emb2)
        
        # Process each block
        for prefix, block_nodes in blocks.items():
            if len(block_nodes) < 2:
                continue
            
            for i, n1 in enumerate(block_nodes):
                if n1.id in merged_away:
                    continue
                    
                for n2 in block_nodes[i + 1:]:
                    if n2.id in merged_away:
                        continue
                    
                    # Check exclusions first
                    if self._should_skip_merge(n1.name, n2.name):
                        continue
                    
                    # Quick string-based check
                    is_candidate, similarity, reason = self._might_be_duplicate(n1, n2)
                    
                    if is_candidate and similarity >= self.similarity_threshold:
                        # High confidence from string matching alone
                        pass
                    elif is_candidate or self._string_similarity(n1.name, n2.name) > 0.5:
                        # Borderline - use precomputed embeddings
                        similarity = fast_semantic_similarity(n1, n2)
                        reason = "semantic_embedding"
                        if similarity == 0.0:
                            continue
                    else:
                        continue
                    
                    if similarity >= self.similarity_threshold:
                        # Verify both nodes still exist in graph
                        if not graph.get_node(n1.id) or not graph.get_node(n2.id):
                            continue

                        # Merge: keep higher confidence node
                        if n1.confidence >= n2.confidence:
                            keep, remove = n1, n2
                        else:
                            keep, remove = n2, n1

                        if verbose:
                            print(f"  Merging '{remove.name}' into '{keep.name}' (sim={similarity:.2f}, {reason})")

                        # Capture edges before merge for potential undo
                        redirected_edges = []
                        for edge in graph.edges():
                            if edge.source_id == remove.id or edge.target_id == remove.id:
                                redirected_edges.append({
                                    "relation": edge.relation,
                                    "was_source": edge.source_id == remove.id,
                                    "other_node": edge.target_id if edge.source_id == remove.id else edge.source_id,
                                    "confidence": edge.confidence,
                                })

                        # Perform merge
                        graph.merge_nodes(keep.id, remove.id)
                        merged_away.add(remove.id)
                        merged.append((keep.name, remove.name))
                        
                        # Log with edge info for undo support
                        self.log.log_merge(keep.name, remove.name, similarity, reason, redirected_edges)
        
        return merged

    def _detect_and_resolve_contradictions(
        self, graph: KnowledgeGraph, verbose: bool = True
    ) -> list[dict]:
        """
        Find contradictory relations and resolve by keeping higher confidence.
        """
        resolved = []
        edges_to_remove = set()  # Track IDs of edges to remove
        processed_pairs = set()  # Track (edge1_id, edge2_id) pairs we've handled
        
        for edge in graph.edges():
            if edge.id in edges_to_remove:
                continue
                
            opposite_rel = OPPOSITE_RELATIONS.get(edge.relation)
            if not opposite_rel:
                continue
            
            # Look for contradicting edge
            conflicts = graph.find_edges(
                source_id=edge.source_id,
                target_id=edge.target_id,
                relation=opposite_rel,
            )
            
            for conflict in conflicts:
                # Skip if we've already handled this pair (in either order)
                pair_key = tuple(sorted([edge.id, conflict.id]))
                if pair_key in processed_pairs:
                    continue
                if conflict.id in edges_to_remove:
                    continue
                    
                processed_pairs.add(pair_key)
                
                # Determine which to keep
                if edge.confidence >= conflict.confidence:
                    keep, remove = edge, conflict
                else:
                    keep, remove = conflict, edge
                
                if verbose:
                    source_node = graph.get_node(keep.source_id)
                    target_node = graph.get_node(keep.target_id)
                    source_name = source_node.name if source_node else keep.source_id
                    target_name = target_node.name if target_node else keep.target_id
                    print(f"  Contradiction: '{source_name}' {keep.relation}/{remove.relation} '{target_name}'")
                    print(f"    Keeping '{keep.relation}' (conf={keep.confidence:.2f}), "
                          f"removing '{remove.relation}' (conf={remove.confidence:.2f})")
                
                # Add disputed_by metadata to surviving edge
                if "disputed_by" not in keep.properties:
                    keep.properties["disputed_by"] = []
                keep.properties["disputed_by"].append({
                    "relation": remove.relation,
                    "confidence": remove.confidence,
                    "source": str(remove.source) if remove.source else None,
                })
                
                # Reduce confidence slightly (epistemic humility)
                keep.confidence = max(0.1, keep.confidence * 0.95)
                
                edges_to_remove.add(remove.id)
                
                resolved.append({
                    "source": keep.source_id,
                    "target": keep.target_id,
                    "kept_relation": keep.relation,
                    "removed_relation": remove.relation,
                })
                
                # Log it
                self.log.log_contradiction(
                    kept_edge={
                        "relation": keep.relation,
                        "confidence": keep.confidence,
                    },
                    removed_edge={
                        "relation": remove.relation,
                        "confidence": remove.confidence,
                    },
                    reason="higher_confidence",
                )
        
        # Remove conflicting edges
        for edge_id in edges_to_remove:
            graph.remove_edge(edge_id)
        
        return resolved

    # Patterns for ephemeral/boilerplate nodes that should always be removed
    EPHEMERAL_PATTERNS = [
        r'^(matrix|vector|scalar|variable|function|point)\s+[A-Za-z]$',  # matrix A, vector x
        r'^[A-Za-z]$',  # single letters: A, B, x, y
        r'^(matrix|vector)\s+[A-Za-z]\d*$',  # matrix A1, vector x2
        r'^f\s*\(\s*[a-z]\s*\)$',  # f(x), f(t)
        r'^[a-z]\s*=\s*',  # x = ..., y = ...
    ]

    BOILERPLATE_TERMS = [
        'copyright', 'cengage', 'rights reserved', 'all rights', 'isbn',
        'electronic rights', 'third party', 'suppressed', 'permissions',
        'trademark', 'reproduced', 'publisher', 'content_suppression',
    ]

    def _is_ephemeral_node(self, name: str) -> bool:
        """Check if this node name represents ephemeral/boilerplate content."""
        import re
        name_lower = name.lower()

        # Check boilerplate terms
        for term in self.BOILERPLATE_TERMS:
            if term in name_lower:
                return True

        # Check ephemeral patterns
        for pattern in self.EPHEMERAL_PATTERNS:
            if re.match(pattern, name, re.IGNORECASE):
                return True

        return False

    def _remove_ephemeral_nodes(self, graph: KnowledgeGraph, verbose: bool = True) -> list[str]:
        """
        Remove ephemeral nodes (variables, boilerplate) regardless of connections.
        """
        removed = []
        nodes_to_remove = []

        for node in graph.nodes():
            if node.name in {"Self", "Thing"}:
                continue

            if self._is_ephemeral_node(node.name):
                if verbose:
                    print(f"  Removing ephemeral: '{node.name}'")
                nodes_to_remove.append(node)
                removed.append(node.name)

                self.log.log_action({
                    "type": "remove_ephemeral",
                    "name": node.name,
                    "reason": "ephemeral_or_boilerplate",
                })

        for node in nodes_to_remove:
            graph.remove_node(node.id)

        return removed

    def _cleanup_orphans(
        self,
        graph: KnowledgeGraph,
        verbose: bool = True,
        merge_threshold: float = 0.85,
        relate_threshold: float = 0.5,
        n: int = 1,
        base_concept_name: str | None = None,
    ) -> dict:
        """
        Find homes for disconnected clusters before pruning.

        Finds isolated subgraphs of size n (or larger) that are disconnected
        from the main graph, then connects them based on semantic similarity:

        - High similarity (>= merge_threshold): Merge into existing node
        - Medium similarity (>= relate_threshold): Create related_to edge
        - Low similarity: Link to base "Thing" concept

        Args:
            graph: The knowledge graph
            verbose: Whether to print progress
            merge_threshold: Minimum similarity to merge (default 0.85)
            relate_threshold: Minimum similarity to create relation (default 0.5)
            n: Minimum cluster size to consider (1=single orphans, 2=pairs, etc.)
            base_concept_name: Name of base concept for fallback linking.
                               If None, auto-detects from graph (concept > Thing > entity).

        Returns:
            Dict with keys:
            - merged: List of (orphan_name, merged_into) tuples
            - related: Count of orphans connected via related_to edges
            - linked_to_base: Count of orphans linked to Thing
        """
        rehomed = []
        skip_names = {"Self", "Thing", "self", "user"}

        # Step 1: Build adjacency map and identify connected components
        node_by_id = {}
        adjacency = {}

        for node in graph.nodes():
            if node.name in skip_names:
                continue
            node_by_id[node.id] = node
            adjacency[node.id] = set()

        for edge in graph.edges():
            if edge.source_id in adjacency and edge.target_id in adjacency:
                adjacency[edge.source_id].add(edge.target_id)
                adjacency[edge.target_id].add(edge.source_id)

        # Step 2: Find connected components using BFS
        visited = set()
        components = []

        for start_id in adjacency:
            if start_id in visited:
                continue

            component = []
            queue = [start_id]
            visited.add(start_id)

            while queue:
                node_id = queue.pop(0)
                component.append(node_id)

                for neighbor_id in adjacency[node_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)

            components.append(component)

        # Step 3: Separate main graph from isolated clusters
        if not components:
            if verbose:
                print("  No nodes to process")
            return rehomed

        components.sort(key=len, reverse=True)
        main_component = set(components[0])
        isolated_clusters = [c for c in components[1:] if len(c) >= n]

        # Also include true orphans (size 1) if n=1
        orphan_ids = [nid for nid in adjacency if not adjacency[nid]]
        if n == 1 and orphan_ids:
            isolated_clusters.extend([[oid] for oid in orphan_ids])

        if not isolated_clusters:
            if verbose:
                print(f"  No isolated clusters of size >= {n} found")
            return rehomed

        total_isolated = sum(len(c) for c in isolated_clusters)
        if verbose:
            print(f"  Found {len(isolated_clusters)} isolated clusters ({total_isolated} nodes), main graph has {len(main_component)} nodes")

        if not main_component:
            if verbose:
                print("  No main graph to merge into")
            return rehomed

        # Step 4: Collect nodes
        isolated_nodes = [node_by_id[nid] for c in isolated_clusters for nid in c if nid in node_by_id]
        main_nodes = [node_by_id[nid] for nid in main_component if nid in node_by_id]

        # Step 5: Precompute embeddings
        all_nodes = isolated_nodes + main_nodes
        embeddings = self._precompute_embeddings(all_nodes, verbose)

        # Step 6: Build embedding matrices for vectorized similarity
        # Filter to nodes with embeddings
        isolated_with_emb = [(n, embeddings[n.id]) for n in isolated_nodes if n.id in embeddings]
        main_with_emb = [(n, embeddings[n.id]) for n in main_nodes if n.id in embeddings]

        if not isolated_with_emb or not main_with_emb:
            if verbose:
                print("  No embeddings available for comparison")
            return rehomed

        if verbose:
            print(f"  Computing {len(isolated_with_emb)} x {len(main_with_emb)} similarity matrix...")

        # Step 7: Compute similarities - use numpy if available for speed
        nodes_to_merge = []  # High similarity - will merge
        nodes_to_relate = []  # Medium similarity - will create edge

        try:
            import numpy as np

            # Build matrices
            iso_matrix = np.array([emb for _, emb in isolated_with_emb])
            main_matrix = np.array([emb for _, emb in main_with_emb])

            # Normalize rows
            iso_norms = np.linalg.norm(iso_matrix, axis=1, keepdims=True)
            main_norms = np.linalg.norm(main_matrix, axis=1, keepdims=True)

            iso_norms[iso_norms == 0] = 1.0
            main_norms[main_norms == 0] = 1.0

            iso_normalized = iso_matrix / iso_norms
            main_normalized = main_matrix / main_norms

            # Compute all cosine similarities at once: (n_iso, n_main)
            similarity_matrix = iso_normalized @ main_normalized.T

            # Find best match for each isolated node
            best_indices = np.argmax(similarity_matrix, axis=1)
            best_similarities = similarity_matrix[np.arange(len(isolated_with_emb)), best_indices]

            for i, (iso_node, _) in enumerate(isolated_with_emb):
                sim = float(best_similarities[i])
                target_node = main_with_emb[best_indices[i]][0]
                if sim >= merge_threshold:
                    nodes_to_merge.append((iso_node, target_node, sim))
                elif sim >= relate_threshold:
                    nodes_to_relate.append((iso_node, target_node, sim))

            if verbose:
                print(f"  (used numpy acceleration)")

        except ImportError:
            # Fallback to pure Python
            main_norms = []
            for _, emb in main_with_emb:
                norm = sum(x * x for x in emb) ** 0.5
                main_norms.append(norm if norm > 0 else 1.0)

            for iso_node, iso_emb in isolated_with_emb:
                iso_norm = sum(x * x for x in iso_emb) ** 0.5
                if iso_norm == 0:
                    continue

                best_idx = -1
                best_similarity = 0.0

                for idx, (main_node, main_emb) in enumerate(main_with_emb):
                    dot = sum(a * b for a, b in zip(iso_emb, main_emb))
                    similarity = dot / (iso_norm * main_norms[idx])

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_idx = idx

                if best_idx >= 0:
                    target_node = main_with_emb[best_idx][0]
                    if best_similarity >= merge_threshold:
                        nodes_to_merge.append((iso_node, target_node, best_similarity))
                    elif best_similarity >= relate_threshold:
                        nodes_to_relate.append((iso_node, target_node, best_similarity))

        # Step 8: Perform merges for high-similarity matches
        merged_ids = set()
        for orphan, target, similarity in nodes_to_merge:
            if not graph.get_node(orphan.id) or not graph.get_node(target.id):
                continue

            if verbose:
                print(f"  Rehoming '{orphan.name}' → '{target.name}' (sim={similarity:.2f})")

            graph.merge_nodes(target.id, orphan.id)
            merged_ids.add(orphan.id)
            rehomed.append((orphan.name, target.name))

            self.log.log_action({
                "type": "rehome_orphan",
                "orphan": orphan.name,
                "merged_into": target.name,
                "similarity": similarity,
            })

        # Step 9: Create related_to edges for medium-similarity matches
        from tiny_mind.substrate.edge import Edge
        from tiny_mind.substrate.source import Source, SourceType
        import uuid

        related_ids = set()
        for orphan, target, similarity in nodes_to_relate:
            if orphan.id in merged_ids:
                continue
            if not graph.get_node(orphan.id) or not graph.get_node(target.id):
                continue

            # Check if edge already exists
            existing = graph.find_edges(source_id=orphan.id, target_id=target.id)
            existing += graph.find_edges(source_id=target.id, target_id=orphan.id)
            if existing:
                related_ids.add(orphan.id)
                continue

            if verbose:
                print(f"  Relating '{orphan.name}' → '{target.name}' (sim={similarity:.2f})")

            new_edge = Edge(
                id=str(uuid.uuid4()),
                source_id=orphan.id,
                target_id=target.id,
                relation="related_to",
                confidence=similarity,
                source=Source(
                    source_type=SourceType.INFERENCE,
                    reference="orphan_relation",
                ),
                properties={"inferred": True, "similarity": similarity},
            )
            graph.add_edge(new_edge)
            related_ids.add(orphan.id)

            self.log.log_action({
                "type": "relate_orphan",
                "orphan": orphan.name,
                "related_to": target.name,
                "similarity": similarity,
            })

        if verbose and related_ids:
            print(f"  Created {len(related_ids)} related_to edges")

        # Step 10: Connect remaining orphans to base concept (Thing or concept)
        # Thing = concrete/physical, concept = abstract
        from tiny_mind.substrate.node import Node as NodeClass

        thing_node = graph.find_node("Thing")
        concept_node = graph.find_node("concept")

        # Create base nodes if they don't exist
        if not thing_node:
            thing_node = NodeClass(
                id=str(uuid.uuid4()),
                name="Thing",
                confidence=1.0,
                source=Source(source_type=SourceType.INFERENCE, reference="base_ontology"),
                properties={"is_category": True, "definition": "Concrete or physical entities"},
            )
            graph.add_node(thing_node)
            if verbose:
                print("  Created base 'Thing' node")

        if not concept_node:
            concept_node = NodeClass(
                id=str(uuid.uuid4()),
                name="concept",
                confidence=1.0,
                source=Source(source_type=SourceType.INFERENCE, reference="base_ontology"),
                properties={"is_category": True, "definition": "Abstract ideas and concepts"},
            )
            graph.add_node(concept_node)
            if verbose:
                print("  Created base 'concept' node")

        # Get embeddings for Thing and concept to classify orphans
        thing_emb = self._get_embedding("physical object, concrete thing, tangible item")
        concept_emb = self._get_embedding("abstract idea, concept, intangible notion")

        linked_count = 0
        thing_count = 0
        concept_count = 0

        for iso_node, iso_emb in isolated_with_emb:
            if iso_node.id in merged_ids or iso_node.id in related_ids:
                continue
            if not graph.get_node(iso_node.id):
                continue

            # Check if already connected
            edges = graph.find_edges(source_id=iso_node.id) + graph.find_edges(target_id=iso_node.id)
            if edges:
                continue

            # Classify as Thing or concept based on embedding similarity
            thing_sim = self._cosine_similarity(iso_emb, thing_emb)
            concept_sim = self._cosine_similarity(iso_emb, concept_emb)

            if thing_sim > concept_sim:
                base_node = thing_node
                thing_count += 1
            else:
                base_node = concept_node
                concept_count += 1

            # Create is_a edge
            new_edge = Edge(
                id=str(uuid.uuid4()),
                source_id=iso_node.id,
                target_id=base_node.id,
                relation="is_a",
                confidence=0.5,
                source=Source(
                    source_type=SourceType.INFERENCE,
                    reference="orphan_linkage",
                ),
                properties={"inferred": True, "reason": "orphan_fallback"},
            )
            graph.add_edge(new_edge)
            linked_count += 1

            self.log.log_action({
                "type": "link_orphan_to_base",
                "orphan": iso_node.name,
                "base": base_node.name,
            })

        if verbose and linked_count > 0:
            print(f"  Linked {linked_count} orphans to base ({thing_count} → Thing, {concept_count} → concept)")

        return {
            "merged": rehomed,
            "related": len(related_ids),
            "linked_to_base": linked_count,
        }

    def deep_audit(
        self,
        graph: KnowledgeGraph,
        verbose: bool = True,
        coherence_threshold: float = 0.4,
        relocation_threshold: float = 0.6,
        min_neighbors: int = 2,
        dry_run: bool = True,
    ) -> list[dict]:
        """
        Deep audit of the entire graph to find misplaced nodes.

        For each node with neighbors, computes how well it "fits" with those
        neighbors using embedding similarity. Nodes with low coherence scores
        are candidates for relocation. Then searches the entire graph for
        potentially better homes.

        This is O(n²) and should only be run periodically.

        Args:
            graph: The knowledge graph
            verbose: Whether to print progress
            coherence_threshold: Nodes with avg neighbor similarity below this
                                 are considered potentially misplaced (default 0.4)
            relocation_threshold: Minimum similarity to suggest relocation (default 0.6)
            min_neighbors: Minimum neighbors to evaluate a node (default 2)
            dry_run: If True, only report suggestions. If False, apply relocations.

        Returns:
            List of dicts describing suggested/applied relocations
        """
        import uuid
        from tiny_mind.substrate.edge import Edge
        from tiny_mind.substrate.source import Source, SourceType

        relocations = []
        skip_names = {"Self", "Thing", "self", "user", "concept", "entity"}

        if verbose:
            print("=" * 50)
            print("  DEEP PLACEMENT AUDIT")
            print("=" * 50)

        # Step 1: Get all nodes and precompute embeddings
        nodes = [n for n in graph.nodes() if n.name not in skip_names]

        if verbose:
            print(f"\n[1/4] Precomputing embeddings for {len(nodes)} nodes...")

        embeddings = self._precompute_embeddings(nodes, verbose)
        node_by_id = {n.id: n for n in nodes}

        # Step 2: Compute coherence score for each node
        if verbose:
            print(f"\n[2/4] Computing coherence scores...")

        coherence_scores = {}
        low_coherence_nodes = []

        for node in nodes:
            if node.id not in embeddings:
                continue

            # Get neighbors
            edges = graph.find_edges(source_id=node.id) + graph.find_edges(target_id=node.id)
            neighbor_ids = set()
            for edge in edges:
                other_id = edge.target_id if edge.source_id == node.id else edge.source_id
                if other_id in embeddings and other_id != node.id:
                    neighbor_ids.add(other_id)

            if len(neighbor_ids) < min_neighbors:
                continue

            # Compute average similarity with neighbors
            node_emb = embeddings[node.id]
            similarities = []
            for nid in neighbor_ids:
                sim = self._cosine_similarity(node_emb, embeddings[nid])
                similarities.append(sim)

            avg_coherence = sum(similarities) / len(similarities)
            coherence_scores[node.id] = avg_coherence

            if avg_coherence < coherence_threshold:
                low_coherence_nodes.append((node, avg_coherence, list(neighbor_ids)))

        if verbose:
            print(f"  Found {len(low_coherence_nodes)} nodes with low coherence (< {coherence_threshold})")

        if not low_coherence_nodes:
            if verbose:
                print("\n  No misplaced nodes detected!")
            return relocations

        # Step 3: For low-coherence nodes, search for better homes
        if verbose:
            print(f"\n[3/4] Searching for better placements...")

        # Build list of all node embeddings for search
        all_node_ids = [nid for nid in embeddings.keys()]

        # Collect raw suggestions first (before filtering)
        raw_suggestions = []

        try:
            import numpy as np

            # Build matrix for fast similarity search
            emb_matrix = np.array([embeddings[nid] for nid in all_node_ids])
            emb_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            emb_norms[emb_norms == 0] = 1.0
            emb_normalized = emb_matrix / emb_norms

            id_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}

            for node, coherence, current_neighbor_ids in low_coherence_nodes:
                if node.id not in id_to_idx:
                    continue

                node_idx = id_to_idx[node.id]
                node_vec = emb_normalized[node_idx:node_idx+1]

                # Compute similarity to all nodes
                similarities = (node_vec @ emb_normalized.T).flatten()

                # Find best matches that aren't current neighbors or self
                current_set = set(current_neighbor_ids) | {node.id}

                best_new_neighbors = []
                for idx in np.argsort(similarities)[::-1]:
                    candidate_id = all_node_ids[idx]
                    if candidate_id in current_set:
                        continue
                    if similarities[idx] >= relocation_threshold:
                        best_new_neighbors.append((candidate_id, float(similarities[idx])))
                    if len(best_new_neighbors) >= 3:
                        break

                if best_new_neighbors:
                    best_id, best_sim = best_new_neighbors[0]
                    best_node = node_by_id.get(best_id)

                    if best_node and best_sim > coherence + 0.15:
                        raw_suggestions.append({
                            "source_node": node,
                            "target_node": best_node,
                            "source_coherence": coherence,
                            "similarity": best_sim,
                        })

        except ImportError:
            # Fallback without numpy
            if verbose:
                print("  (numpy not available, using slower method)")

            for node, coherence, current_neighbor_ids in low_coherence_nodes:
                node_emb = embeddings.get(node.id)
                if not node_emb:
                    continue

                current_set = set(current_neighbor_ids) | {node.id}
                best_match = None
                best_sim = 0.0

                for other_id, other_emb in embeddings.items():
                    if other_id in current_set:
                        continue
                    sim = self._cosine_similarity(node_emb, other_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = node_by_id.get(other_id)

                if best_match and best_sim >= relocation_threshold and best_sim > coherence + 0.15:
                    raw_suggestions.append({
                        "source_node": node,
                        "target_node": best_match,
                        "source_coherence": coherence,
                        "similarity": best_sim,
                    })

        # Step 3b: Filter and normalize suggestions
        # - Detect circular pairs (A→B and B→A) → treat as duplicates
        # - Always have longer name merge into shorter name

        # Build set of suggestion pairs for circular detection
        suggestion_pairs = {}
        for s in raw_suggestions:
            key = (s["source_node"].id, s["target_node"].id)
            suggestion_pairs[key] = s

        circular_pairs = set()
        for s in raw_suggestions:
            src_id = s["source_node"].id
            tgt_id = s["target_node"].id
            reverse_key = (tgt_id, src_id)
            if reverse_key in suggestion_pairs:
                # Circular - both point at each other
                circular_pairs.add(tuple(sorted([src_id, tgt_id])))

        # Process suggestions
        seen_pairs = set()
        for s in raw_suggestions:
            src = s["source_node"]
            tgt = s["target_node"]
            pair_key = tuple(sorted([src.id, tgt.id]))

            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            is_circular = pair_key in circular_pairs
            sim = s["similarity"]

            # Normalize: longer name should merge into shorter name
            if len(src.name) < len(tgt.name):
                # Swap - target has longer name, should merge into source
                src, tgt = tgt, src

            # Only report high-similarity matches as merge candidates
            # Circular pairs only count if similarity is also high
            # Skip loose relations - embedding similarity != should be connected
            if sim < 0.80:
                continue

            suggestion_type = "duplicate"
            action = "merge"

            relocation = {
                "node": src.name,
                "target": tgt.name,
                "similarity": round(sim, 3),
                "type": suggestion_type,
                "action": action,
                "applied": False,
            }

            if not dry_run:
                # Merge longer into shorter
                src_in_graph = graph.get_node(src.id)
                tgt_in_graph = graph.get_node(tgt.id)
                if src_in_graph and tgt_in_graph:
                    graph.merge_nodes(tgt.id, src.id)
                    relocation["applied"] = True
                    self.log.log_action({
                        "type": "audit_merge",
                        "merged": src.name,
                        "into": tgt.name,
                        "similarity": sim,
                    })

            relocations.append(relocation)

            if verbose:
                status = "→ applied" if relocation["applied"] else "(dry run)"
                print(f"  '{src.name}' → '{tgt.name}' (sim={sim:.2f}) {status}")

        # Step 4: Summary
        if verbose:
            print(f"\n[4/4] Summary")
            print("=" * 50)
            if relocations:
                applied = sum(1 for r in relocations if r["applied"])
                print(f"  Found {len(relocations)} merge candidates")
                if dry_run:
                    print(f"  (dry run - no changes applied)")
                else:
                    print(f"  Applied {applied} changes")
            else:
                print(f"  No relocation candidates found")

        # Save cache
        self._save_embedding_cache()

        return relocations

    def cluster_by_embeddings(
        self,
        graph: KnowledgeGraph,
        verbose: bool = True,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        assign_noise: bool = True,
        n_clusters: int = None,
        use_hdbscan: bool = False,
    ) -> dict[str, int]:
        """
        Cluster nodes by embedding similarity.

        Uses silhouette score to find optimal k for KMeans by default.
        If n_clusters is specified, uses that fixed k.
        If use_hdbscan=True, uses HDBSCAN for automatic detection.

        Args:
            graph: The knowledge graph
            verbose: Whether to print progress
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN core points
            assign_noise: If True, assign unclustered nodes to nearest cluster
            n_clusters: If set, use KMeans with this many clusters
            use_hdbscan: If True, use HDBSCAN instead of KMeans

        Returns:
            Mapping of node_id -> cluster_id (-1 for noise/unclustered)
        """
        skip_names = {"Self", "Thing", "self", "user", "concept", "entity"}

        if verbose:
            print("=" * 50)
            print("  EMBEDDING-BASED CLUSTERING")
            print("=" * 50)

        # Step 1: Get all nodes and precompute embeddings
        nodes = [n for n in graph.nodes() if n.name not in skip_names]

        if verbose:
            print(f"\n[1/4] Precomputing embeddings for {len(nodes)} nodes...")

        embeddings = self._precompute_embeddings(nodes, verbose)
        node_by_id = {n.id: n for n in nodes}

        # Filter to nodes with embeddings
        nodes_with_emb = [(n, embeddings[n.id]) for n in nodes if n.id in embeddings]

        if len(nodes_with_emb) < min_cluster_size:
            if verbose:
                print(f"  Not enough nodes for clustering ({len(nodes_with_emb)} < {min_cluster_size})")
            return {}

        import numpy as np
        node_ids = [n.id for n, _ in nodes_with_emb]
        emb_matrix = np.array([emb for _, emb in nodes_with_emb])

        if use_hdbscan:
            # Use HDBSCAN for automatic cluster detection
            if verbose:
                print(f"\n[2/4] Running HDBSCAN clustering...")
            
            try:
                from hdbscan import HDBSCAN
            except ImportError:
                if verbose:
                    print("  HDBSCAN not available. Install with: pip install hdbscan")
                return {}

            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
            )
            labels = clusterer.fit_predict(emb_matrix)

            # Count initial results
            initial_noise = sum(1 for l in labels if l == -1)
            if verbose:
                print(f"  Initial: {len(set(labels)) - (1 if -1 in labels else 0)} clusters, {initial_noise} noise")

        else:
            # Use KMeans
            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
            except ImportError:
                if verbose:
                    print("  sklearn not available. Install with: pip install scikit-learn")
                return {}

            if n_clusters is not None:
                # Fixed k specified
                if verbose:
                    print(f"\n[2/4] Running KMeans clustering (k={n_clusters})...")
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(emb_matrix)
                
                if verbose:
                    print(f"  Created {n_clusters} clusters")
            else:
                # Find optimal k using silhouette score
                if verbose:
                    print(f"\n[2/4] Finding optimal k using silhouette score...")
                
                n_samples = len(emb_matrix)
                # Search range: 2 to min(sqrt(n)*2, 30, n-1)
                max_k = min(int(np.sqrt(n_samples) * 2), 30, n_samples - 1)
                min_k = 2
                
                if max_k < min_k:
                    max_k = min_k
                
                best_k = min_k
                best_score = -1
                scores = []
                
                if verbose:
                    print(f"  Testing k from {min_k} to {max_k}...")
                
                for k in range(min_k, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels_k = kmeans.fit_predict(emb_matrix)
                    score = silhouette_score(emb_matrix, labels_k)
                    scores.append((k, score))
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                
                if verbose:
                    print(f"  Best k={best_k} (silhouette={best_score:.3f})")
                    # Show top 5 k values
                    top_scores = sorted(scores, key=lambda x: -x[1])[:5]
                    print(f"  Top candidates: {', '.join(f'k={k}({s:.2f})' for k, s in top_scores)}")
                
                # Run final clustering with best k
                kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(emb_matrix)
            
            initial_noise = 0  # KMeans assigns all points

        # Step 3: Assign noise points to nearest cluster (HDBSCAN only)
        if assign_noise and initial_noise > 0:
            if verbose:
                print(f"\n[3/4] Assigning {initial_noise} noise points to nearest clusters...")

            # Compute cluster centroids
            unique_clusters = [c for c in set(labels) if c >= 0]
            if unique_clusters:
                cluster_centroids = {}
                for cid in unique_clusters:
                    mask = labels == cid
                    cluster_centroids[cid] = emb_matrix[mask].mean(axis=0)

                # For each noise point, find nearest centroid
                centroid_matrix = np.array([cluster_centroids[cid] for cid in unique_clusters])
                centroid_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
                centroid_norms[centroid_norms == 0] = 1.0
                centroid_normalized = centroid_matrix / centroid_norms

                for i, label in enumerate(labels):
                    if label == -1:
                        # Find nearest cluster by cosine similarity
                        node_emb = emb_matrix[i:i+1]
                        node_norm = np.linalg.norm(node_emb)
                        if node_norm > 0:
                            node_normalized = node_emb / node_norm
                            similarities = (node_normalized @ centroid_normalized.T).flatten()
                            best_cluster_idx = np.argmax(similarities)
                            labels[i] = unique_clusters[best_cluster_idx]

        # Step 4: Store cluster assignments in node properties
        if verbose:
            print(f"\n[4/4] Storing cluster assignments...")

        node_to_cluster = {}
        cluster_counts = {}

        for i, node_id in enumerate(node_ids):
            cluster_id = int(labels[i])
            node_to_cluster[node_id] = cluster_id

            # Update node properties
            node = graph.get_node(node_id)
            if node:
                if node.properties is None:
                    node.properties = {}
                node.properties["cluster_id"] = cluster_id

            # Count cluster sizes
            if cluster_id not in cluster_counts:
                cluster_counts[cluster_id] = 0
            cluster_counts[cluster_id] += 1

        # Summary
        if verbose:
            num_clusters = len([c for c in cluster_counts.keys() if c >= 0])
            noise_count = cluster_counts.get(-1, 0)
            print(f"\n  Final: {num_clusters} clusters")
            if noise_count > 0:
                print(f"  Remaining noise: {noise_count} nodes")

            # Show largest clusters
            sorted_clusters = sorted(
                [(cid, cnt) for cid, cnt in cluster_counts.items() if cid >= 0],
                key=lambda x: -x[1]
            )
            print(f"\n  Largest clusters:")
            for cid, cnt in sorted_clusters[:10]:
                print(f"    Cluster {cid}: {cnt} nodes")
            if len(sorted_clusters) > 10:
                print(f"    ... and {len(sorted_clusters) - 10} more")

        # Save embedding cache
        self._save_embedding_cache()

        return node_to_cluster

    def _prune(self, graph: KnowledgeGraph, verbose: bool = True) -> list[str]:
        """
        Remove low-confidence orphan nodes that are stale.
        """
        pruned = []
        nodes_to_remove = []

        for node in graph.nodes():
            # Skip meta nodes
            if node.name in {"Self", "Thing"}:
                continue

            # Check if orphan (no connections)
            neighbors = list(graph.get_neighbors(node.id))
            if neighbors:
                continue

            # Check confidence and staleness
            staleness = node.staleness_days() if hasattr(node, "staleness_days") else 0

            if (node.confidence < self.prune_confidence_threshold and
                staleness > self.prune_staleness_days):

                if verbose:
                    print(f"  Pruning orphan '{node.name}' (conf={node.confidence:.2f}, stale={staleness}d)")

                nodes_to_remove.append(node)
                pruned.append(node.name)

                self.log.log_prune(
                    node.name,
                    node.confidence,
                    f"orphan, low_confidence ({node.confidence:.2f}), stale ({staleness}d)",
                )
        
        for node in nodes_to_remove:
            graph.remove_node(node.id)
        
        return pruned

    def _infer_hierarchies(
        self, 
        graph: KnowledgeGraph, 
        verbose: bool = True,
        max_candidates: int = 50,
        similarity_threshold: float = 0.5,
    ) -> list[dict]:
        """
        Infer missing hierarchical relationships between unconnected nodes.
        
        Uses embeddings to find semantically similar but unconnected nodes,
        then uses LLM to determine if hierarchical relationships exist.
        
        Optimized with blocking to avoid O(n²) comparisons.
        """
        import uuid
        from collections import defaultdict
        from tiny_mind.substrate.edge import Edge
        from tiny_mind.substrate.source import Source, SourceType
        from tiny_mind.extraction.prompts import HIERARCHY_INFERENCE_PROMPT
        from tiny_mind.extraction.schemas import HierarchyInferenceSchema
        
        inferred = []
        nodes = list(graph.nodes())
        
        # Skip if too few nodes
        if len(nodes) < 2:
            return inferred
        
        # Skip meta nodes
        skip_names = {"Self", "Thing"}
        nodes = [n for n in nodes if n.name not in skip_names]
        
        if verbose:
            print(f"  Analyzing {len(nodes)} nodes for missing hierarchies...")
        
        # Find existing connections (to avoid suggesting duplicates)
        # Use node IDs for faster lookup
        existing_connections = set()
        for edge in graph.edges():
            existing_connections.add((edge.source_id, edge.target_id))
            existing_connections.add((edge.target_id, edge.source_id))
        
        # Build a map of node_id -> node for quick lookup
        node_by_id = {n.id: n for n in nodes}
        
        # Use blocking to reduce comparisons - only compare nodes that might be related
        # Block by: common words, similar length, or category markers
        blocks: dict[str, list] = defaultdict(list)
        
        category_markers = {'is_category', 'category', 'type', 'class', 'kind'}
        
        for node in nodes:
            name_lower = node.name.lower()
            words = name_lower.split()
            
            # Block by each significant word (length > 2)
            for word in words:
                if len(word) > 2:
                    blocks[word].append(node)
            
            # Block by first 4 chars of name
            if len(name_lower) >= 4:
                blocks[f"_prefix_{name_lower[:4]}"].append(node)
            
            # Block category nodes together (they're likely parents)
            if any(marker in node.properties for marker in category_markers):
                blocks["_categories"].append(node)
            if node.properties.get("is_category"):
                blocks["_categories"].append(node)
        
        if verbose:
            total_block_pairs = sum(len(b) * (len(b) - 1) // 2 for b in blocks.values())
            print(f"  Using {len(blocks)} blocks, ~{total_block_pairs} candidate comparisons (vs {len(nodes)*(len(nodes)-1)//2} full)")
        
        # Precompute embeddings only for nodes we'll actually compare
        nodes_to_embed = set()
        for block_nodes in blocks.values():
            if len(block_nodes) >= 2:
                for n in block_nodes:
                    nodes_to_embed.add(n.id)
        
        nodes_for_embedding = [n for n in nodes if n.id in nodes_to_embed]
        if not nodes_for_embedding:
            if verbose:
                print("  No candidate pairs found for hierarchy inference")
            return inferred
            
        node_embeddings = self._precompute_embeddings(nodes_for_embedding, verbose)
        
        # Find candidate pairs within blocks
        candidates = []
        seen_pairs = set()
        
        for block_name, block_nodes in blocks.items():
            if len(block_nodes) < 2:
                continue
                
            for i, n1 in enumerate(block_nodes):
                if n1.id not in node_embeddings:
                    continue
                    
                for n2 in block_nodes[i + 1:]:
                    if n2.id not in node_embeddings:
                        continue
                    
                    # Skip if we've seen this pair
                    pair_key = tuple(sorted([n1.id, n2.id]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    
                    # Skip if already connected
                    if (n1.id, n2.id) in existing_connections:
                        continue
                    
                    emb1 = node_embeddings[n1.id]
                    emb2 = node_embeddings[n2.id]
                    similarity = self._cosine_similarity(emb1, emb2)
                    
                    if similarity >= similarity_threshold:
                        candidates.append({
                            "node1": n1.name,
                            "node2": n2.name,
                            "node1_id": n1.id,
                            "node2_id": n2.id,
                            "similarity": similarity,
                        })
        
        if not candidates:
            if verbose:
                print("  No candidate pairs found for hierarchy inference")
            return inferred
        
        # Sort by similarity and take top candidates
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        candidates = candidates[:max_candidates]
        
        if verbose:
            print(f"  Found {len(candidates)} candidate pairs for LLM analysis")
        
        # Format candidates for LLM
        pairs_str = "\n".join([
            f"- {c['node1']} <-> {c['node2']} (similarity: {c['similarity']:.2f})"
            for c in candidates
        ])
        
        # Call LLM to infer relationships
        try:
            if self.llm_provider == "openai":
                from openai import OpenAI
                client = OpenAI()
                
                response = client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You analyze knowledge graphs to find missing hierarchical relationships."},
                        {"role": "user", "content": HIERARCHY_INFERENCE_PROMPT.format(candidate_pairs=pairs_str)},
                    ],
                    temperature=0.3,
                    response_format=HierarchyInferenceSchema,
                )
                result = response.choices[0].message.parsed
            else:
                if verbose:
                    print("  Hierarchy inference only supported with OpenAI provider")
                return inferred
                
        except Exception as e:
            if verbose:
                print(f"  Hierarchy inference failed: {e}")
            return inferred
        
        # Create edges for inferred relations
        for rel in result.inferred_relations:
            source_node = graph.find_node(rel.source_name)
            target_node = graph.find_node(rel.target_name)
            
            if not source_node or not target_node:
                continue
            
            # Check if this edge already exists
            existing = graph.find_edges(
                source_id=source_node.id,
                target_id=target_node.id,
                relation=rel.relation,
            )
            if existing:
                continue
            
            if verbose:
                print(f"  Inferring: {rel.source_name} {rel.relation} {rel.target_name} ({rel.confidence:.2f})")
            
            new_edge = Edge(
                id=str(uuid.uuid4()),
                source_id=source_node.id,
                target_id=target_node.id,
                relation=rel.relation,
                confidence=rel.confidence * 0.8,
                source=Source(
                    source_type=SourceType.INFERENCE,
                    reference="hierarchy_inference",
                    raw_content=rel.reasoning,
                ),
                properties={"inferred": True, "reasoning": rel.reasoning},
            )
            
            graph.add_edge(new_edge)
            inferred.append({
                "source": rel.source_name,
                "target": rel.target_name,
                "relation": rel.relation,
                "confidence": rel.confidence,
            })
            
            self.log.log_action({
                "type": "infer_hierarchy",
                "source": rel.source_name,
                "target": rel.target_name,
                "relation": rel.relation,
                "confidence": rel.confidence,
                "reasoning": rel.reasoning,
            })
        
        return inferred

    def _detect_and_split_confused_nodes(
        self,
        graph: KnowledgeGraph,
        verbose: bool = True,
        min_edges: int = 4,
        cluster_separation_threshold: float = 0.3,
    ) -> list[dict]:
        """
        Detect nodes with incoherent connections and split them.

        Uses embeddings to detect when a node's neighbors form distinct,
        well-separated clusters - a signal that the node represents multiple
        different concepts that got conflated.

        This is reactive (based on actual confusion) rather than proactive
        (based on theoretical polysemy).

        Args:
            graph: The knowledge graph to modify
            verbose: Whether to print progress
            min_edges: Minimum edges to consider (default 4)
            cluster_separation_threshold: How different clusters must be (0-1)

        Returns:
            List of dicts describing each split performed
        """
        import uuid
        import re
        from tiny_mind.substrate.node import Node
        from tiny_mind.substrate.source import Source, SourceType

        splits_performed = []

        # Skip meta nodes and already-qualified nodes
        skip_names = {"Self", "Thing", "self", "user"}
        qualifier_pattern = r'^(.+?)\s*\(([^)]+)\)$'

        # Find candidate nodes
        candidates = []
        for node in graph.nodes():
            if node.name in skip_names:
                continue
            if re.match(qualifier_pattern, node.name):
                continue

            edges = graph.find_edges(source_id=node.id) + graph.find_edges(target_id=node.id)
            if len(edges) < min_edges:
                continue

            candidates.append((node, edges))

        if verbose:
            print(f"  Checking {len(candidates)} nodes for confusion...")

        if not candidates:
            return splits_performed

        # Process each candidate
        for node, edges in candidates:
            # Step 1: Get connected nodes
            connected = []
            for edge in edges:
                other_id = edge.target_id if edge.source_id == node.id else edge.source_id
                other_node = graph.get_node(other_id)
                if other_node and other_node.name not in skip_names:
                    connected.append((edge, other_node))

            if len(connected) < min_edges:
                continue

            # Step 2: Get embeddings for all connected nodes
            neighbor_embeddings = {}
            for edge, other_node in connected:
                emb = self._get_embedding(other_node.name)
                neighbor_embeddings[other_node.id] = (other_node, edge, emb)

            if len(neighbor_embeddings) < min_edges:
                continue

            # Step 3: Compute pairwise similarity matrix
            node_ids = list(neighbor_embeddings.keys())
            n = len(node_ids)
            similarity_matrix = [[0.0] * n for _ in range(n)]

            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        similarity_matrix[i][j] = 1.0
                    else:
                        emb_i = neighbor_embeddings[node_ids[i]][2]
                        emb_j = neighbor_embeddings[node_ids[j]][2]
                        sim = self._cosine_similarity(emb_i, emb_j)
                        similarity_matrix[i][j] = sim
                        similarity_matrix[j][i] = sim

            # Step 4: Simple clustering - find well-separated groups
            # Use a greedy approach: start with most dissimilar pair, grow clusters
            clusters = self._find_clusters(similarity_matrix, cluster_separation_threshold)

            if len(clusters) < 2:
                continue  # No confusion detected

            # Check cluster sizes - each must have at least 2 nodes
            valid_clusters = [c for c in clusters if len(c) >= 2]
            if len(valid_clusters) < 2:
                continue

            if verbose:
                print(f"  '{node.name}': detected {len(valid_clusters)} distinct clusters")

            # Step 5: Name clusters by their most representative node
            cluster_info = []
            for cluster_indices in valid_clusters:
                cluster_nodes = [neighbor_embeddings[node_ids[i]] for i in cluster_indices]
                # Pick the node with shortest name as representative (often most general)
                representative = min(cluster_nodes, key=lambda x: len(x[0].name))
                cluster_name = representative[0].name.lower()
                # Clean up the name for use as qualifier
                cluster_name = re.sub(r'[^\w\s]', '', cluster_name).strip()
                if len(cluster_name) > 30:
                    cluster_name = cluster_name[:30].rsplit(' ', 1)[0]
                cluster_info.append((cluster_name, cluster_nodes))

            if verbose:
                for name, nodes in cluster_info:
                    node_names = [n[0].name for n in nodes[:3]]
                    print(f"    Cluster '{name}': {len(nodes)} nodes ({', '.join(node_names)}...)")

            # Step 6: Create split nodes
            new_nodes = []
            for cluster_name, cluster_nodes in cluster_info:
                new_name = f"{node.name} ({cluster_name})"

                existing = graph.find_node(new_name)
                if existing:
                    new_nodes.append((cluster_name, existing, cluster_nodes, False))
                    continue

                new_node = Node(
                    id=str(uuid.uuid4()),
                    name=new_name,
                    properties={
                        **node.properties,
                        "split_from": node.name,
                        "cluster_representative": cluster_name,
                    },
                    confidence=node.confidence,
                    source=Source(
                        source_type=SourceType.INFERENCE,
                        reference="confusion_split",
                        raw_content=f"Split from confused node '{node.name}' based on cluster around '{cluster_name}'",
                    ),
                )
                graph.add_node(new_node)
                new_nodes.append((cluster_name, new_node, cluster_nodes, True))

            # Step 7: Reassign edges
            edges_reassigned = 0
            for cluster_name, new_node, cluster_nodes, was_created in new_nodes:
                for other_node, edge, emb in cluster_nodes:
                    if edge.source_id == node.id:
                        edge.source_id = new_node.id
                    else:
                        edge.target_id = new_node.id
                    edges_reassigned += 1

            # Remove original if no edges remain
            remaining = graph.find_edges(source_id=node.id) + graph.find_edges(target_id=node.id)
            if not remaining:
                graph.remove_node(node.id)
                if verbose:
                    print(f"    Removed original '{node.name}'")
            else:
                if verbose:
                    print(f"    Kept original '{node.name}' with {len(remaining)} edges")

            split_info = {
                "original_name": node.name,
                "new_nodes": [
                    {"name": n.name, "cluster": c, "edges": len(nodes)}
                    for c, n, nodes, _ in new_nodes
                ],
                "edges_reassigned": edges_reassigned,
            }
            splits_performed.append(split_info)

            self.log.log_action({
                "type": "confusion_split",
                "original": node.name,
                "splits": [n.name for _, n, _, _ in new_nodes],
                "cluster_names": [c for c, _, _, _ in new_nodes],
            })

        return splits_performed

    def _find_clusters(
        self,
        similarity_matrix: list[list[float]],
        separation_threshold: float = 0.3,
    ) -> list[list[int]]:
        """
        Find well-separated clusters in a similarity matrix.

        Uses a simple approach:
        1. Build a graph where nodes are connected if similarity > threshold
        2. Find connected components
        3. Return clusters that are well-separated from each other

        Args:
            similarity_matrix: NxN similarity matrix
            separation_threshold: Similarity below this = different clusters

        Returns:
            List of clusters, where each cluster is a list of indices
        """
        n = len(similarity_matrix)
        if n < 2:
            return [[0]] if n == 1 else []

        # Compute average similarity to determine adaptive threshold
        total_sim = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += similarity_matrix[i][j]
                count += 1
        avg_sim = total_sim / count if count > 0 else 0.5

        # Use threshold relative to average - items below average are "different"
        connect_threshold = avg_sim - separation_threshold

        # Build adjacency list
        adjacency = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] >= connect_threshold:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        # Find connected components using BFS
        visited = [False] * n
        clusters = []

        for start in range(n):
            if visited[start]:
                continue

            # BFS from this node
            cluster = []
            queue = [start]
            visited[start] = True

            while queue:
                node = queue.pop(0)
                cluster.append(node)

                for neighbor in adjacency[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            clusters.append(cluster)

        return clusters

    def _extract_base_and_qualifier(self, name: str) -> tuple[str, str | None]:
        """
        Extract base name and qualifier from a node name.

        Returns (base, qualifier) or (name, None) if no qualifier.
        Handles stacked qualifiers by extracting the innermost base.
        Ignores function notation like f(x).
        """
        import re
        # Pattern: name (qualifier) - but qualifier must be 3+ chars to avoid f(x)
        pattern = r'^(.+?)\s*\(([^)]{3,})\)$'

        # Keep stripping qualifiers to get base (handles stacked)
        base = name
        last_qualifier = None

        while True:
            match = re.match(pattern, base)
            if match:
                base = match.group(1).strip()
                last_qualifier = match.group(2).strip()
            else:
                break

        # If we found any qualifier, return the base and last qualifier found
        if base != name:
            return base, last_qualifier
        return name, None

    def _is_mathematical_notation(self, text: str) -> bool:
        """
        Check if parenthetical content is mathematical notation, not a qualifier.

        Examples of notation (not qualifiers):
        - point (1, 3) → coordinates
        - f(x) → function application
        - sin(θ) → trig function
        - matrix (2x3) → dimensions
        """
        import re

        # Contains numbers → likely coordinates or dimensions
        if re.search(r'\d', text):
            return True

        # Contains comma → likely a tuple/coordinates
        if ',' in text:
            return True

        # Single letter or very short → likely a variable
        if len(text) <= 2:
            return True

        # Looks like a variable (single lowercase letter with optional subscript)
        if re.match(r'^[a-z][0-9]*$', text):
            return True

        return False

    def _is_valid_domain_qualifier(self, qualifier: str) -> bool:
        """
        Check if a qualifier is a valid domain name (not an LLM description
        and not mathematical notation).

        Valid: mathematics, biology, programming, physics, trigonometry
        Invalid: mathematical_function, abstract description, 1, 3, x
        """
        # Mathematical notation - not a qualifier at all
        if self._is_mathematical_notation(qualifier):
            return True  # Return True to KEEP it (it's not a qualifier to strip)

        # Contains underscore = LLM formatting
        if '_' in qualifier:
            return False

        # Must be a single word - domains are one word
        if ' ' in qualifier:
            return False

        # Very long = probably not a domain
        if len(qualifier) > 20:
            return False

        return True

    def _cleanup_qualifiers(
        self,
        graph: KnowledgeGraph,
        verbose: bool = True
    ) -> list[tuple[str, str]]:
        """
        Clean up qualifiers in three passes:

        Pass 0: Strip ALL stacked qualifiers - reduce to base or base + one qualifier.

        Pass 1: Strip all INVALID qualifiers (LLM descriptions with underscores,
                multi-word phrases, etc.) unconditionally.

        Pass 2: For remaining VALID qualifiers, only keep if there are 2+
                different qualifiers for the same base (actual disambiguation).
        """
        from collections import defaultdict
        import re

        cleaned = []

        # === PASS 0: Strip stacked qualifiers ===
        # Any node with 2+ qualifiers gets reduced to base name
        for node in list(graph.nodes()):
            # Count parenthetical expressions
            paren_count = node.name.count('(')
            if paren_count < 2:
                continue  # Not stacked

            # Has stacked qualifiers - extract base
            base, _ = self._extract_base_and_qualifier(node.name)
            if base == node.name or base is None:
                continue

            old_name = node.name
            existing_base = graph.find_node(base)

            if existing_base and existing_base.id != node.id:
                if verbose:
                    print(f"  Stacked: '{old_name}' → merge into '{base}'")
                graph.merge_nodes(existing_base.id, node.id)
            else:
                if verbose:
                    print(f"  Stacked: '{old_name}' → '{base}'")
                node.name = base

            cleaned.append((old_name, base))
            self.log.log_action({
                "type": "cleanup_stacked",
                "original": old_name,
                "normalized": base,
            })

        # === PASS 1: Strip invalid qualifiers unconditionally ===
        nodes_with_qualifiers = 0

        for node in list(graph.nodes()):
            base, qualifier = self._extract_base_and_qualifier(node.name)

            if qualifier is None:
                continue

            nodes_with_qualifiers += 1

            # Check if qualifier is valid (real domain name)
            is_valid = self._is_valid_domain_qualifier(qualifier)

            if is_valid:
                continue  # Keep for pass 2

            # Invalid qualifier - strip it
            old_name = node.name
            existing_base = graph.find_node(base)

            if existing_base and existing_base.id != node.id:
                if verbose:
                    print(f"  Invalid qualifier: '{old_name}' → merge into '{base}'")
                graph.merge_nodes(existing_base.id, node.id)
            else:
                if verbose:
                    print(f"  Invalid qualifier: '{old_name}' → '{base}'")
                node.name = base

            cleaned.append((old_name, base))
            self.log.log_action({
                "type": "cleanup_invalid_qualifier",
                "original": old_name,
                "normalized": base,
                "removed": qualifier,
            })

        # === PASS 2: Strip ALL qualifiers - merge into base name ===
        # The philosophy: "time is time" regardless of who studies it.
        # Domain qualifiers (physics, sociology, etc.) don't represent different things.
        # If confusion detection later finds the node is genuinely polysemous, it will split.
        base_to_nodes: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
        nodes_by_name: dict[str, object] = {}

        for node in graph.nodes():
            nodes_by_name[node.name] = node
            base, qualifier = self._extract_base_and_qualifier(node.name)
            base_to_nodes[base.lower()].append((node.name, qualifier))

        # For each base, merge all qualified versions into the base
        for base_lower, variants in base_to_nodes.items():
            # Find the canonical base name (prefer existing unqualified node)
            unqualified = [name for name, q in variants if q is None]
            if unqualified:
                canonical_base = unqualified[0]
            else:
                # Extract true base by stripping ALL qualifiers
                canonical_base, _ = self._extract_base_and_qualifier(variants[0][0])
                if canonical_base is None:
                    canonical_base = variants[0][0]

            # Strip ALL qualifiers - merge everything into base
            for name, qualifier in variants:
                if qualifier is None:
                    continue  # Already unqualified

                node = nodes_by_name.get(name)
                if not node:
                    continue

                old_name = node.name
                existing_base = graph.find_node(canonical_base)

                if existing_base and existing_base.id != node.id:
                    if verbose:
                        print(f"  Merging: '{old_name}' → '{canonical_base}'")
                    graph.merge_nodes(existing_base.id, node.id)
                else:
                    if verbose:
                        print(f"  Renaming: '{old_name}' → '{canonical_base}'")
                    node.name = canonical_base

                cleaned.append((old_name, canonical_base))
                self.log.log_action({
                    "type": "cleanup_qualifier",
                    "original": old_name,
                    "normalized": canonical_base,
                })

        return cleaned

    def _undo_bad_merges(self, graph: KnowledgeGraph, verbose: bool = True) -> list[tuple[str, str]]:
        """
        Check revision log for merges that violated current rules and undo them.

        Creates new nodes for concepts that were incorrectly merged.
        Restores edges that were redirected during the original merge.
        Only processes each bad merge once - tracks via undo_merge log entries.
        """
        import uuid
        from tiny_mind.substrate.node import Node
        from tiny_mind.substrate.edge import Edge
        from tiny_mind.substrate.source import Source, SourceType

        undone = []

        # Read the log
        log_entries = self.log.read_log()
        merges = [e for e in log_entries if e.get('type') == 'merge']
        
        # Find all undos that have already been processed
        already_undone = set()
        for e in log_entries:
            if e.get('type') == 'undo_merge':
                # Track by (recreated, was_merged_into) pair
                key = (e.get('recreated', ''), e.get('was_merged_into', ''))
                already_undone.add(key)

        for merge in merges:
            kept_name = merge.get('kept', '')
            removed_name = merge.get('removed', '')
            
            # Check if we've already undone this specific merge
            if (removed_name, kept_name) in already_undone:
                continue

            # Check if this merge would now be skipped
            if self._should_skip_merge(kept_name, removed_name):
                # This was a bad merge - check if the kept node still exists
                kept_node = graph.find_node(kept_name)
                if not kept_node:
                    continue

                # Check if removed concept already exists (already fixed)
                if graph.find_node(removed_name):
                    continue

                if verbose:
                    print(f"  Undoing bad merge: recreating '{removed_name}' (was merged into '{kept_name}')")

                # Create new node for the removed concept
                new_node = Node(
                    id=str(uuid.uuid4()),
                    name=removed_name,
                    confidence=0.5,  # Moderate confidence since we're recreating
                    source=Source(
                        source_type=SourceType.INFERENCE,
                        reference="split_from_bad_merge",
                        raw_content=f"Recreated from bad merge with '{kept_name}'",
                    ),
                    properties={"split_from": kept_name},
                )

                graph.add_node(new_node)
                
                # Restore edges that were redirected during the merge
                redirected_edges = merge.get('redirected_edges', [])
                edges_restored = 0
                for edge_info in redirected_edges:
                    other_node_id = edge_info.get('other_node', '')
                    # Check if the other node still exists
                    if not graph.get_node(other_node_id):
                        continue
                        
                    # Create edge pointing to/from the recreated node
                    if edge_info.get('was_source', False):
                        source_id, target_id = new_node.id, other_node_id
                    else:
                        source_id, target_id = other_node_id, new_node.id
                    
                    new_edge = Edge(
                        id=str(uuid.uuid4()),
                        source_id=source_id,
                        target_id=target_id,
                        relation=edge_info.get('relation', 'related_to'),
                        confidence=edge_info.get('confidence', 0.5) * 0.8,  # Slightly reduce confidence
                        source=Source(
                            source_type=SourceType.INFERENCE,
                            reference="restored_from_undo_merge",
                        ),
                    )
                    graph.add_edge(new_edge)
                    edges_restored += 1
                
                if verbose and edges_restored > 0:
                    print(f"    Restored {edges_restored} edges")
                
                undone.append((kept_name, removed_name))

                # Log the undo - this prevents future re-processing
                self.log.log_action({
                    "type": "undo_merge",
                    "recreated": removed_name,
                    "was_merged_into": kept_name,
                    "reason": "violated_merge_rules",
                    "edges_restored": edges_restored,
                })

        return undone

    def revise(
        self,
        graph: KnowledgeGraph,
        verbose: bool = True,
        orphan_cluster_size: int = 1,
    ) -> RevisionResult:
        """
        Run full maintenance pass on knowledge graph.

        All actions are auto-accepted and logged.

        Args:
            graph: The knowledge graph to revise
            verbose: Whether to print progress
            orphan_cluster_size: Minimum cluster size for orphan rehoming
                                 (1=single orphans, 2=pairs, etc.)
        """
        result = RevisionResult()

        if verbose:
            print("=" * 50)
            print("  KNOWLEDGE REVISION")
            print("=" * 50)

        # 0. Learn domain relationships from graph structure
        if verbose:
            print("\n[0/9] Learning domain relationships from graph...")
        self._learn_domain_relationships(graph, verbose)

        # 1. Clean up problematic qualifiers (stacked and unnecessary)
        if verbose:
            print("\n[1/9] Cleaning up qualifiers...")
        qualifiers_cleaned = self._cleanup_qualifiers(graph, verbose)
        if verbose and qualifiers_cleaned:
            print(f"  Cleaned {len(qualifiers_cleaned)} problematic qualifiers")

        # 2. Undo bad merges - DISABLED
        # New philosophy: merge everything, let confusion detection split if needed
        # The old undo logic was recreating qualified nodes we just cleaned up
        if verbose:
            print("\n[2/9] Checking for bad merges to undo... (skipped - using confusion detection instead)")
        result.undone_merges = []

        # 3. Remove ephemeral/boilerplate nodes (before dedup to avoid wasted merges)
        if verbose:
            print("\n[3/9] Removing ephemeral/boilerplate nodes...")
        result.ephemeral_removed = self._remove_ephemeral_nodes(graph, verbose)

        # 4. Deduplication
        if verbose:
            print("\n[4/9] Deduplication...")
        result.merged_nodes = self._deduplicate(graph, verbose)

        # 5. Infer missing hierarchical relationships
        if verbose:
            print("\n[5/9] Inferring missing hierarchies...")
        result.inferred_relations = self._infer_hierarchies(graph, verbose)

        # 6. Detect and split confused nodes (based on cluster separation)
        if verbose:
            print("\n[6/9] Detecting confused nodes (cluster analysis)...")
        result.split_nodes = self._detect_and_split_confused_nodes(graph, verbose)

        # 7. Contradiction resolution
        if verbose:
            print("\n[7/9] Contradiction detection...")
        result.contradictions_resolved = self._detect_and_resolve_contradictions(graph, verbose)

        # 8. Cleanup orphans - find homes for disconnected nodes
        if verbose:
            print("\n[8/9] Rehoming orphan nodes...")
        orphan_result = self._cleanup_orphans(graph, verbose, n=orphan_cluster_size)
        result.orphans_rehomed = orphan_result["merged"]
        result.orphans_related = orphan_result["related"]
        result.orphans_linked_to_base = orphan_result["linked_to_base"]

        # 9. Pruning low-confidence orphans
        if verbose:
            print("\n[9/9] Pruning low-value orphan nodes...")
        result.pruned = self._prune(graph, verbose)

        # Save embedding cache
        self._save_embedding_cache()

        if verbose:
            print("\n" + "=" * 50)
            print(result.summary())
            print("=" * 50)

        return result
