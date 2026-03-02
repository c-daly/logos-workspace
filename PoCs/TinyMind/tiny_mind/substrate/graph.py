"""
Knowledge Graph - the mind's memory structure.

This is the in-memory graph that holds all knowledge. It provides
operations for querying, modifying, and reasoning about the graph.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterator, Optional
import json

from .node import Node, TemporalGrain, create_self
from .edge import Edge, Relations
from .source import Source, SourceType


@dataclass
class GraphStats:
    """Statistics about the knowledge graph."""
    node_count: int = 0
    edge_count: int = 0
    relation_types: dict[str, int] = field(default_factory=dict)
    orphan_count: int = 0  # Nodes with no edges
    avg_confidence: float = 0.0
    oldest_node_days: float = 0.0
    most_accessed_node: Optional[str] = None


class KnowledgeGraph:
    """
    The mind's knowledge structure.

    Stores nodes and edges, provides querying and modification operations.
    This is the in-memory version; can be persisted to Neo4j.
    """

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}

        # Indexes for fast lookup
        self._edges_by_source: dict[str, list[str]] = {}  # source_id -> [edge_ids]
        self._edges_by_target: dict[str, list[str]] = {}  # target_id -> [edge_ids]
        self._edges_by_relation: dict[str, list[str]] = {}  # relation -> [edge_ids]
        self._nodes_by_name: dict[str, list[str]] = {}  # name -> [node_ids]
        self._nodes_by_prefix: dict[str, set[str]] = {}  # 3-char prefix -> node_ids
        self._nodes_by_trigram: dict[str, set[str]] = {}  # trigram -> node_ids

        # Bootstrap with Self node
        self._bootstrap()

    def _bootstrap(self):
        """Initialize with minimal seed knowledge."""
        # The agent knows it exists
        self_node = create_self()
        self.add_node(self_node)

        # Create some fundamental concepts
        thing = Node(
            id="thing",
            name="Thing",
            properties={"is_root_concept": True},
            source=Source.bootstrap("The most general category"),
            confidence=1.0,
            temporal_grain=TemporalGrain.ETERNAL,
        )
        self.add_node(thing)

        # Self is a Thing
        self.add_edge(Edge(
            source_id="self",
            target_id="thing",
            relation=Relations.IS_A,
            source=Source.bootstrap("Self is a thing"),
            confidence=1.0,
        ))

    # === Node Operations ===

    def add_node(self, node: Node) -> Node:
        """Add a node to the graph."""
        self._nodes[node.id] = node
        self._index_node(node)
        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        node = self._nodes.get(node_id)
        if node:
            node.touch()
        return node

    def get_nodes_by_name(self, name: str) -> list[Node]:
        """Get all nodes with a given name."""
        node_ids = self._nodes_by_name.get(name.lower(), [])
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def find_node(self, name: str) -> Optional[Node]:
        """Find a single node by name (returns first match)."""
        nodes = self.get_nodes_by_name(name)
        return nodes[0] if nodes else None

    def search_nodes(self, query: str, limit: int = 10) -> list[Node]:
        """Search for nodes whose name contains the query (case-insensitive)."""
        query_lower = query.lower()
        matches = []
        for node in self._nodes.values():
            if query_lower in node.name.lower():
                matches.append(node)
                if len(matches) >= limit:
                    break
        return matches

    def find_similar_nodes(
        self, 
        name: str, 
        threshold: float = 0.5,
        limit: int = 10
    ) -> list[tuple[Node, float]]:
        """
        Find nodes with similar names using trigram index.
        
        Returns list of (node, similarity_score) tuples, sorted by score.
        Uses Jaccard similarity on trigrams for fast approximate matching.
        """
        from difflib import SequenceMatcher
        
        name_lower = name.lower().strip()
        query_trigrams = self._get_trigrams(name_lower)
        
        if not query_trigrams:
            return []
        
        # Find candidate nodes that share at least one trigram
        candidate_ids: set[str] = set()
        for trigram in query_trigrams:
            if trigram in self._nodes_by_trigram:
                candidate_ids.update(self._nodes_by_trigram[trigram])
        
        # Score candidates using full string similarity
        results = []
        for node_id in candidate_ids:
            node = self._nodes.get(node_id)
            if not node:
                continue
            
            node_name_lower = node.name.lower()
            
            # Quick Jaccard similarity on trigrams for filtering
            node_trigrams = self._get_trigrams(node_name_lower)
            jaccard = len(query_trigrams & node_trigrams) / len(query_trigrams | node_trigrams)
            
            if jaccard < threshold * 0.5:  # Quick filter
                continue
            
            # Full string similarity for final score
            score = SequenceMatcher(None, name_lower, node_name_lower).ratio()
            
            if score >= threshold:
                results.append((node, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            return False

        # Remove all edges connected to this node
        edge_ids_to_remove = (
            self._edges_by_source.get(node_id, []) +
            self._edges_by_target.get(node_id, [])
        )
        for edge_id in set(edge_ids_to_remove):
            self.remove_edge(edge_id)

        # Remove from indexes
        node = self._nodes[node_id]
        self._unindex_node(node)

        # Remove node
        del self._nodes[node_id]
        return True

    def update_node(self, node_id: str, updates: dict) -> Optional[Node]:
        """Update properties of a node."""
        node = self.get_node(node_id)
        if not node:
            return None

        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)
            else:
                node.properties[key] = value

        return node

    # === Edge Operations ===

    def add_edge(self, edge: Edge) -> Edge:
        """Add an edge to the graph."""
        # Validate that both nodes exist
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node {edge.source_id} does not exist")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node {edge.target_id} does not exist")

        self._edges[edge.id] = edge
        self._index_edge(edge)
        return edge

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID."""
        edge = self._edges.get(edge_id)
        if edge:
            edge.touch()
        return edge

    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge."""
        if edge_id not in self._edges:
            return False

        edge = self._edges[edge_id]
        self._unindex_edge(edge)
        del self._edges[edge_id]
        return True

    def find_edges(
        self,
        source_id: str = None,
        target_id: str = None,
        relation: str = None,
    ) -> list[Edge]:
        """Find edges matching criteria."""
        candidates = set(self._edges.keys())

        if source_id:
            candidates &= set(self._edges_by_source.get(source_id, []))
        if target_id:
            candidates &= set(self._edges_by_target.get(target_id, []))
        if relation:
            candidates &= set(self._edges_by_relation.get(relation, []))

        return [self._edges[eid] for eid in candidates]

    # === Query Operations ===

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",  # "out", "in", "both"
        relation: str = None,
    ) -> list[Node]:
        """Get neighboring nodes."""
        neighbors = []

        if direction in ("out", "both"):
            for edge_id in self._edges_by_source.get(node_id, []):
                edge = self._edges.get(edge_id)
                if edge and (relation is None or edge.relation == relation):
                    neighbor = self.get_node(edge.target_id)
                    if neighbor:
                        neighbors.append(neighbor)

        if direction in ("in", "both"):
            for edge_id in self._edges_by_target.get(node_id, []):
                edge = self._edges.get(edge_id)
                if edge and (relation is None or edge.relation == relation):
                    neighbor = self.get_node(edge.source_id)
                    if neighbor:
                        neighbors.append(neighbor)

        return neighbors

    def get_types(self, node_id: str) -> list[Node]:
        """Get all types this node is an instance of (via is_a edges)."""
        return self.get_neighbors(node_id, direction="out", relation=Relations.IS_A)

    def get_instances(self, type_id: str) -> list[Node]:
        """Get all instances of this type."""
        return self.get_neighbors(type_id, direction="in", relation=Relations.IS_A)

    def get_causes(self, node_id: str) -> list[Node]:
        """What causes this node?"""
        return self.get_neighbors(node_id, direction="in", relation=Relations.CAUSES)

    def get_effects(self, node_id: str) -> list[Node]:
        """What does this node cause?"""
        return self.get_neighbors(node_id, direction="out", relation=Relations.CAUSES)

    def get_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 5,
    ) -> Optional[list[tuple[Node, Edge]]]:
        """Find a path between two nodes (BFS)."""
        if from_id == to_id:
            return []

        visited = {from_id}
        queue = [(from_id, [])]

        while queue:
            current_id, path = queue.pop(0)

            for edge_id in self._edges_by_source.get(current_id, []):
                edge = self._edges.get(edge_id)
                if not edge:
                    continue

                next_id = edge.target_id
                if next_id == to_id:
                    return path + [(self._nodes[next_id], edge)]

                if next_id not in visited and len(path) < max_depth:
                    visited.add(next_id)
                    queue.append((next_id, path + [(self._nodes[next_id], edge)]))

        return None

    def get_orphans(self) -> list[Node]:
        """Get nodes with no connections."""
        orphans = []
        for node_id, node in self._nodes.items():
            has_edges = (
                self._edges_by_source.get(node_id, []) or
                self._edges_by_target.get(node_id, [])
            )
            if not has_edges:
                orphans.append(node)
        return orphans

    def get_highly_connected(self, top_n: int = 10) -> list[tuple[Node, int]]:
        """Get the most connected nodes."""
        connectivity = []
        for node_id, node in self._nodes.items():
            edge_count = (
                len(self._edges_by_source.get(node_id, [])) +
                len(self._edges_by_target.get(node_id, []))
            )
            connectivity.append((node, edge_count))

        connectivity.sort(key=lambda x: x[1], reverse=True)
        return connectivity[:top_n]

    # === Modification Operations ===

    def merge_nodes(self, node_id_a: str, node_id_b: str, new_name: str = None) -> Node:
        """Merge two nodes that are discovered to be the same thing."""
        node_a = self.get_node(node_id_a)
        node_b = self.get_node(node_id_b)

        if not node_a or not node_b:
            raise ValueError("Both nodes must exist")

        # Merge properties into node_a
        node_a.merge_properties(node_b)
        if new_name:
            node_a.name = new_name

        # Redirect all edges from node_b to node_a
        for edge_id in list(self._edges_by_source.get(node_id_b, [])):
            edge = self._edges.get(edge_id)
            if edge:
                self._unindex_edge(edge)
                edge.source_id = node_id_a
                self._index_edge(edge)

        for edge_id in list(self._edges_by_target.get(node_id_b, [])):
            edge = self._edges.get(edge_id)
            if edge:
                self._unindex_edge(edge)
                edge.target_id = node_id_a
                self._index_edge(edge)

        # Remove the merged node
        self.remove_node(node_id_b)

        return node_a

    def prune_unused(self, max_staleness_days: float = 30, min_access_count: int = 2) -> list[str]:
        """Remove nodes that haven't been useful."""
        pruned = []
        for node_id in list(self._nodes.keys()):
            node = self._nodes[node_id]
            # Don't prune special nodes
            if node_id in ("self", "thing"):
                continue
            if node.should_prune(max_staleness_days, min_access_count):
                self.remove_node(node_id)
                pruned.append(node_id)
        return pruned

    def decay_associations(self, rate: float = 0.95):
        """Apply decay to weighted association edges."""
        for edge in self._edges.values():
            if edge.should_decay():
                edge.decay_strength(rate)

    # === Statistics ===

    def get_stats(self) -> GraphStats:
        """Get statistics about the graph."""
        stats = GraphStats()
        stats.node_count = len(self._nodes)
        stats.edge_count = len(self._edges)

        # Count relation types
        for edge in self._edges.values():
            stats.relation_types[edge.relation] = stats.relation_types.get(edge.relation, 0) + 1

        # Count orphans
        stats.orphan_count = len(self.get_orphans())

        # Average confidence
        if self._nodes:
            stats.avg_confidence = sum(n.confidence for n in self._nodes.values()) / len(self._nodes)

        # Oldest node
        if self._nodes:
            oldest = max(self._nodes.values(), key=lambda n: n.age_days())
            stats.oldest_node_days = oldest.age_days()

        # Most accessed
        if self._nodes:
            most_accessed = max(self._nodes.values(), key=lambda n: n.access_count)
            stats.most_accessed_node = most_accessed.name

        return stats

    # === Indexing ===

    def _get_trigrams(self, text: str) -> set[str]:
        """Extract trigrams from text for fuzzy indexing."""
        text = text.lower().strip()
        if len(text) < 3:
            return {text} if text else set()
        return {text[i:i+3] for i in range(len(text) - 2)}

    def _index_node(self, node: Node):
        """Add node to indexes."""
        name_key = node.name.lower()
        if name_key not in self._nodes_by_name:
            self._nodes_by_name[name_key] = []
        if node.id not in self._nodes_by_name[name_key]:
            self._nodes_by_name[name_key].append(node.id)
        
        # Prefix index (first 3 chars)
        if len(name_key) >= 3:
            prefix = name_key[:3]
            if prefix not in self._nodes_by_prefix:
                self._nodes_by_prefix[prefix] = set()
            self._nodes_by_prefix[prefix].add(node.id)
        
        # Trigram index for fuzzy matching
        for trigram in self._get_trigrams(name_key):
            if trigram not in self._nodes_by_trigram:
                self._nodes_by_trigram[trigram] = set()
            self._nodes_by_trigram[trigram].add(node.id)

    def _unindex_node(self, node: Node):
        """Remove node from indexes."""
        name_key = node.name.lower()
        if name_key in self._nodes_by_name:
            if node.id in self._nodes_by_name[name_key]:
                self._nodes_by_name[name_key].remove(node.id)
        
        # Remove from prefix index
        if len(name_key) >= 3:
            prefix = name_key[:3]
            if prefix in self._nodes_by_prefix:
                self._nodes_by_prefix[prefix].discard(node.id)
        
        # Remove from trigram index
        for trigram in self._get_trigrams(name_key):
            if trigram in self._nodes_by_trigram:
                self._nodes_by_trigram[trigram].discard(node.id)

    def _index_edge(self, edge: Edge):
        """Add edge to indexes."""
        if edge.source_id not in self._edges_by_source:
            self._edges_by_source[edge.source_id] = []
        if edge.id not in self._edges_by_source[edge.source_id]:
            self._edges_by_source[edge.source_id].append(edge.id)

        if edge.target_id not in self._edges_by_target:
            self._edges_by_target[edge.target_id] = []
        if edge.id not in self._edges_by_target[edge.target_id]:
            self._edges_by_target[edge.target_id].append(edge.id)

        if edge.relation not in self._edges_by_relation:
            self._edges_by_relation[edge.relation] = []
        if edge.id not in self._edges_by_relation[edge.relation]:
            self._edges_by_relation[edge.relation].append(edge.id)

    def _unindex_edge(self, edge: Edge):
        """Remove edge from indexes."""
        if edge.source_id in self._edges_by_source:
            if edge.id in self._edges_by_source[edge.source_id]:
                self._edges_by_source[edge.source_id].remove(edge.id)

        if edge.target_id in self._edges_by_target:
            if edge.id in self._edges_by_target[edge.target_id]:
                self._edges_by_target[edge.target_id].remove(edge.id)

        if edge.relation in self._edges_by_relation:
            if edge.id in self._edges_by_relation[edge.relation]:
                self._edges_by_relation[edge.relation].remove(edge.id)

    # === Serialization ===

    def to_dict(self) -> dict:
        """Serialize the entire graph."""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
            "edges": {eid: edge.to_dict() for eid, edge in self._edges.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """Deserialize a graph."""
        graph = cls.__new__(cls)
        graph._nodes = {}
        graph._edges = {}
        graph._edges_by_source = {}
        graph._edges_by_target = {}
        graph._edges_by_relation = {}
        graph._nodes_by_name = {}
        graph._nodes_by_prefix = {}
        graph._nodes_by_trigram = {}

        # Load nodes
        for nid, node_data in data.get("nodes", {}).items():
            node = Node.from_dict(node_data)
            graph._nodes[nid] = node
            graph._index_node(node)

        # Load edges
        for eid, edge_data in data.get("edges", {}).items():
            edge = Edge.from_dict(edge_data)
            graph._edges[eid] = edge
            graph._index_edge(edge)

        return graph

    def save_to_file(self, filepath: str):
        """Save graph to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "KnowledgeGraph":
        """Load graph from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    # === Iteration ===

    def nodes(self) -> Iterator[Node]:
        """Iterate over all nodes."""
        return iter(self._nodes.values())

    def edges(self) -> Iterator[Edge]:
        """Iterate over all edges."""
        return iter(self._edges.values())

    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self._nodes
