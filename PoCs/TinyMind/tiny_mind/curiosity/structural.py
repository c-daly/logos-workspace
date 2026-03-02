"""
Structural analysis for anomaly detection.

Analyzes the knowledge graph structure to identify:
- Suspicious claims (high confidence, low connectivity)
- Cluster outliers
- Type pattern violations
- Isolated high-importance nodes
"""

from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

from tiny_mind.substrate.graph import KnowledgeGraph
from tiny_mind.substrate.node import Node
from tiny_mind.substrate.edge import Edge


@dataclass
class AnomalyScore:
    """Score indicating how anomalous a node is."""
    node_id: str
    node_name: str
    score: float  # 0-1, higher = more anomalous
    reasons: list[str]
    
    def __str__(self) -> str:
        return f"{self.node_name}: {self.score:.2f} ({', '.join(self.reasons)})"


class StructuralAnalyzer:
    """Analyzes graph structure to detect anomalies and patterns."""
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self._relation_patterns: dict[str, dict] = {}
        self._node_clusters: dict[str, int] = {}
        self._rebuild_caches()
    
    def _rebuild_caches(self):
        """Rebuild internal caches for analysis."""
        self._build_relation_patterns()
        self._build_clusters()
    
    def _build_relation_patterns(self):
        """Build statistics about typical relation patterns."""
        # Track what types of nodes typically appear as source/target for each relation
        patterns = defaultdict(lambda: {"sources": defaultdict(int), "targets": defaultdict(int)})
        
        for edge in self.graph.edges():
            source = self.graph.get_node(edge.source_id)
            target = self.graph.get_node(edge.target_id)
            
            if source and target:
                source_type = source.properties.get("is_a", "unknown")
                target_type = target.properties.get("is_a", "unknown")
                
                # Handle case where is_a is a list (convert to string for dict key)
                if isinstance(source_type, list):
                    source_type = source_type[0] if source_type else "unknown"
                if isinstance(target_type, list):
                    target_type = target_type[0] if target_type else "unknown"
                
                patterns[edge.relation]["sources"][source_type] += 1
                patterns[edge.relation]["targets"][target_type] += 1
        
        self._relation_patterns = dict(patterns)
    
    def _build_clusters(self):
        """Simple clustering based on connectivity."""
        # Use connected component labeling
        visited = set()
        cluster_id = 0
        
        for node in self.graph.nodes():
            if node.id not in visited:
                # BFS to find connected component
                queue = [node.id]
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    self._node_clusters[current] = cluster_id
                    
                    for neighbor in self.graph.get_neighbors(current):
                        if neighbor.id not in visited:
                            queue.append(neighbor.id)
                
                cluster_id += 1
    
    def connectivity_score(self, node: Node) -> float:
        """
        Calculate connectivity score (0-1).
        Higher = more connected.
        """
        neighbors = list(self.graph.get_neighbors(node.id))
        if not neighbors:
            return 0.0
        
        # Normalize by typical connectivity
        all_connections = [len(list(self.graph.get_neighbors(n.id))) for n in self.graph.nodes()]
        if not all_connections:
            return 0.0
            
        max_connections = max(all_connections) if all_connections else 1
        return min(1.0, len(neighbors) / max(max_connections, 1))
    
    def confidence_connectivity_ratio(self, node: Node) -> float:
        """
        High confidence + low connectivity = suspicious.
        Returns anomaly score (0-1), higher = more suspicious.
        """
        connectivity = self.connectivity_score(node)
        confidence = node.confidence
        
        # If high confidence but low connectivity, that's suspicious
        if confidence > 0.7 and connectivity < 0.2:
            return 0.8 + (confidence - 0.7) * 0.5  # Scale to 0.8-0.95
        elif confidence > 0.5 and connectivity < 0.1:
            return 0.6 + (confidence - 0.5) * 0.4
        elif connectivity == 0 and confidence > 0.3:
            return 0.5
        
        return 0.0
    
    def cluster_coherence(self, node: Node) -> float:
        """
        Does this node fit its cluster?
        Returns anomaly score (0-1), higher = less coherent.
        """
        node_cluster = self._node_clusters.get(node.id)
        if node_cluster is None:
            return 0.5  # Unknown cluster
        
        # Get node's type (handle list case)
        node_type = node.properties.get("is_a", "unknown")
        if isinstance(node_type, list):
            node_type = node_type[0] if node_type else "unknown"
        
        # Count types in same cluster
        cluster_types = defaultdict(int)
        for other_id, other_cluster in self._node_clusters.items():
            if other_cluster == node_cluster:
                other_node = self.graph.get_node(other_id)
                if other_node:
                    other_type = other_node.properties.get("is_a", "unknown")
                    if isinstance(other_type, list):
                        other_type = other_type[0] if other_type else "unknown"
                    cluster_types[other_type] += 1
        
        # If this node's type is rare in its cluster, it might be misplaced
        if not cluster_types:
            return 0.0
            
        total = sum(cluster_types.values())
        my_type_count = cluster_types.get(node_type, 0)
        
        if my_type_count / total < 0.1:  # Less than 10% of cluster
            return 0.6
        
        return 0.0
    
    def type_pattern_match(self, edge: Edge) -> float:
        """
        Does this relation match typical patterns?
        Returns anomaly score (0-1), higher = less typical.
        """
        pattern = self._relation_patterns.get(edge.relation)
        if not pattern:
            return 0.0  # No pattern established
        
        source = self.graph.get_node(edge.source_id)
        target = self.graph.get_node(edge.target_id)
        
        if not source or not target:
            return 0.0
        
        source_type = source.properties.get("is_a", "unknown")
        target_type = target.properties.get("is_a", "unknown")
        # Handle list case
        if isinstance(source_type, list):
            source_type = source_type[0] if source_type else "unknown"
        if isinstance(target_type, list):
            target_type = target_type[0] if target_type else "unknown"
        
        # Check if this source/target type is common for this relation
        source_counts = pattern["sources"]
        target_counts = pattern["targets"]
        
        total_sources = sum(source_counts.values())
        total_targets = sum(target_counts.values())
        
        if total_sources == 0 or total_targets == 0:
            return 0.0
        
        source_freq = source_counts.get(source_type, 0) / total_sources
        target_freq = target_counts.get(target_type, 0) / total_targets
        
        # If both are rare, this is an unusual pattern
        if source_freq < 0.05 and target_freq < 0.05:
            return 0.7
        elif source_freq < 0.1 or target_freq < 0.1:
            return 0.4
        
        return 0.0
    
    def has_definition(self, node: Node) -> bool:
        """Check if node has a definition property."""
        return bool(node.properties.get("definition"))
    
    def source_diversity(self, node: Node) -> float:
        """
        How many different sources corroborate this node?
        Returns score (0-1), higher = more diverse (better).
        """
        # For now, just check if there's a source and confirmation count
        if not node.source:
            return 0.0
        
        confirmations = node.source.confirmation_count
        if confirmations >= 3:
            return 1.0
        elif confirmations >= 1:
            return 0.6
        else:
            return 0.3
    
    def calculate_anomaly_score(self, node: Node) -> AnomalyScore:
        """Calculate overall anomaly score for a node."""
        reasons = []
        scores = []
        
        # Check confidence/connectivity ratio
        cc_ratio = self.confidence_connectivity_ratio(node)
        if cc_ratio > 0.3:
            reasons.append(f"high_conf_low_conn ({cc_ratio:.2f})")
            scores.append(cc_ratio)
        
        # Check cluster coherence
        coherence = self.cluster_coherence(node)
        if coherence > 0.3:
            reasons.append(f"cluster_outlier ({coherence:.2f})")
            scores.append(coherence)
        
        # Check source diversity
        diversity = self.source_diversity(node)
        if diversity < 0.5:
            reasons.append(f"low_source_diversity ({diversity:.2f})")
            scores.append(0.5 - diversity)
        
        # Calculate overall score
        if scores:
            overall = sum(scores) / len(scores)
        else:
            overall = 0.0
        
        return AnomalyScore(
            node_id=node.id,
            node_name=node.name,
            score=overall,
            reasons=reasons,
        )
    
    def find_anomalies(self, threshold: float = 0.3, limit: int = 10) -> list[AnomalyScore]:
        """Find nodes that are structurally anomalous."""
        anomalies = []
        
        for node in self.graph.nodes():
            # Skip meta nodes
            if node.name in {"Self", "Thing"}:
                continue
            
            score = self.calculate_anomaly_score(node)
            if score.score >= threshold:
                anomalies.append(score)
        
        # Sort by score descending
        anomalies.sort(key=lambda x: x.score, reverse=True)
        return anomalies[:limit]
    
    def find_missing_definitions(self, min_connections: int = 3) -> list[Node]:
        """Find important nodes (high connectivity) without definitions."""
        missing = []
        
        for node in self.graph.nodes():
            if node.name in {"Self", "Thing"}:
                continue
            
            connections = len(list(self.graph.get_neighbors(node.id)))
            if connections >= min_connections and not self.has_definition(node):
                missing.append(node)
        
        # Sort by connection count
        missing.sort(key=lambda n: len(list(self.graph.get_neighbors(n.id))), reverse=True)
        return missing
    
    def find_isolated_clusters(self, min_size: int = 3) -> list[list[Node]]:
        """Find clusters that are disconnected from the main graph."""
        # Count nodes per cluster
        cluster_sizes = defaultdict(int)
        for cluster_id in self._node_clusters.values():
            cluster_sizes[cluster_id] += 1
        
        # Find the main cluster (largest)
        if not cluster_sizes:
            return []
        
        main_cluster = max(cluster_sizes.keys(), key=lambda k: cluster_sizes[k])
        
        # Find isolated clusters
        isolated = []
        for cluster_id, size in cluster_sizes.items():
            if cluster_id != main_cluster and size >= min_size:
                nodes = [
                    self.graph.get_node(node_id)
                    for node_id, c_id in self._node_clusters.items()
                    if c_id == cluster_id
                ]
                nodes = [n for n in nodes if n is not None]
                if nodes:
                    isolated.append(nodes)
        
        return isolated
    
    def find_shallow_concepts(self, min_connections: int = 5, max_properties: int = 2) -> list[Node]:
        """Find highly-referenced concepts with shallow representation."""
        shallow = []
        
        for node in self.graph.nodes():
            if node.name in {"Self", "Thing"}:
                continue
            
            connections = len(list(self.graph.get_neighbors(node.id)))
            # Count meaningful properties (exclude meta properties)
            meaningful_props = {
                k: v for k, v in node.properties.items()
                if k not in {"auto_created", "from_relation"}
            }
            
            if connections >= min_connections and len(meaningful_props) <= max_properties:
                shallow.append(node)
        
        shallow.sort(key=lambda n: len(list(self.graph.get_neighbors(n.id))), reverse=True)
        return shallow
