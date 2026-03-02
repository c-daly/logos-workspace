"""
Curiosity drive - generates goals from graph state.

Analyzes the knowledge graph to identify what TinyMind should be curious about.
"""

from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional

from tiny_mind.substrate.graph import KnowledgeGraph
from tiny_mind.substrate.node import Node

from .goals import CuriosityGoal, GoalType, GOAL_TYPE_WEIGHTS
from .structural import StructuralAnalyzer


class CuriosityDrive:
    """
    Generates curiosity goals from the current state of the knowledge graph.
    
    Identifies:
    - Gaps in knowledge (missing definitions)
    - Potential connections between concepts
    - Uncertain or suspicious claims
    - Shallow concepts that need depth
    - Anomalous structures
    """
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.analyzer = StructuralAnalyzer(graph)
        
        # Track recently investigated targets to avoid repetition
        self._recently_investigated: dict[str, datetime] = {}
        self._investigation_cooldown_hours = 24
    
    def generate_goals(self, limit: int = 10) -> list[CuriosityGoal]:
        """
        Generate prioritized curiosity goals from current graph state.
        
        Returns goals sorted by priority (highest first).
        """
        # Refresh structural analysis
        self.analyzer._rebuild_caches()
        
        goals = []
        
        # Find different types of goals
        goals.extend(self._find_gaps())
        goals.extend(self._find_connections())
        goals.extend(self._find_uncertainties())
        goals.extend(self._find_depth_opportunities())
        goals.extend(self._find_verification_targets())
        
        # Filter out recently investigated
        goals = [g for g in goals if not self._recently_investigated_check(g.target)]
        
        # Sort by priority
        goals.sort(key=lambda g: g.priority, reverse=True)
        
        return goals[:limit]
    
    def _recently_investigated_check(self, target: str) -> bool:
        """Check if target was recently investigated."""
        last_time = self._recently_investigated.get(target)
        if last_time is None:
            return False
        
        hours_ago = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600
        return hours_ago < self._investigation_cooldown_hours
    
    def mark_investigated(self, target: str):
        """Mark a target as recently investigated."""
        self._recently_investigated[target] = datetime.now(timezone.utc)
    
    def _calculate_importance(self, node: Node) -> float:
        """Calculate how important a node is (based on connectivity)."""
        neighbors = list(self.graph.get_neighbors(node.id))
        
        # Normalize by graph size
        total_nodes = len(list(self.graph.nodes()))
        if total_nodes == 0:
            return 0.0
        
        # More connections = more important
        connection_score = min(1.0, len(neighbors) / max(total_nodes * 0.1, 1))
        
        return connection_score
    
    def _find_gaps(self) -> list[CuriosityGoal]:
        """Find knowledge gaps - important concepts without definitions."""
        goals = []
        
        # Find nodes referenced often but missing definitions
        missing_defs = self.analyzer.find_missing_definitions(min_connections=3)
        
        for node in missing_defs[:5]:  # Limit to top 5
            importance = self._calculate_importance(node)
            connections = len(list(self.graph.get_neighbors(node.id)))
            
            priority = GOAL_TYPE_WEIGHTS[GoalType.GAP] * (0.5 + importance * 0.5)
            
            goals.append(CuriosityGoal(
                type=GoalType.GAP,
                target=node.name,
                question=f"What is the definition of '{node.name}'?",
                priority=priority,
                context={
                    "node_id": node.id,
                    "connections": connections,
                    "reason": "Referenced often but no definition",
                },
                related_nodes=[node.id],
            ))
        
        return goals
    
    def _find_connections(self) -> list[CuriosityGoal]:
        """Find potential connections between disconnected concepts."""
        goals = []
        
        # Find isolated clusters that might connect to main graph
        isolated_clusters = self.analyzer.find_isolated_clusters(min_size=2)
        
        for cluster in isolated_clusters[:3]:  # Top 3 clusters
            # Get most connected node in cluster as representative
            cluster.sort(key=lambda n: len(list(self.graph.get_neighbors(n.id))), reverse=True)
            representative = cluster[0]
            
            # Find potential connection targets in main graph
            # (nodes with similar types or names)
            rep_type = representative.properties.get("is_a", "concept")
            
            goals.append(CuriosityGoal(
                type=GoalType.CONNECTION,
                target=representative.name,
                question=f"How does '{representative.name}' relate to other concepts?",
                priority=GOAL_TYPE_WEIGHTS[GoalType.CONNECTION] * 0.8,
                context={
                    "cluster_size": len(cluster),
                    "cluster_nodes": [n.name for n in cluster],
                    "representative_type": rep_type,
                },
                related_nodes=[n.id for n in cluster],
            ))
        
        # Also look for highly connected nodes in same domain but not directly linked
        self._find_domain_connections(goals)
        
        return goals
    
    def _find_domain_connections(self, goals: list):
        """Find nodes in same domain that should probably be connected."""
        # Group nodes by their is_a type
        by_type = defaultdict(list)
        for node in self.graph.nodes():
            if node.name in {"Self", "Thing"}:
                continue
            node_type = node.properties.get("is_a", "unknown")
            # Handle case where is_a is a list
            if isinstance(node_type, list):
                node_type = node_type[0] if node_type else "unknown"
            if node_type != "unknown":
                by_type[node_type].append(node)
        
        # Look for pairs that have same type but no direct edge
        for type_name, nodes in by_type.items():
            if len(nodes) < 2:
                continue
            
            # Check pairs of important nodes
            important = sorted(nodes, key=lambda n: len(list(self.graph.get_neighbors(n.id))), reverse=True)[:5]
            
            for i, n1 in enumerate(important):
                for n2 in important[i+1:]:
                    # Check if they're directly connected
                    edges = self.graph.find_edges(source_id=n1.id, target_id=n2.id)
                    reverse = self.graph.find_edges(source_id=n2.id, target_id=n1.id)
                    
                    if not edges and not reverse:
                        goals.append(CuriosityGoal(
                            type=GoalType.CONNECTION,
                            target=n1.name,
                            question=f"How does '{n1.name}' relate to '{n2.name}'?",
                            priority=GOAL_TYPE_WEIGHTS[GoalType.CONNECTION] * 0.6,
                            context={
                                "concepts": [n1.name, n2.name],
                                "shared_type": type_name,
                            },
                            related_nodes=[n1.id, n2.id],
                        ))
    
    def _find_uncertainties(self) -> list[CuriosityGoal]:
        """Find low-confidence claims worth verifying."""
        goals = []
        
        # Find nodes with low confidence but some connections
        for node in self.graph.nodes():
            if node.name in {"Self", "Thing"}:
                continue
            
            connections = len(list(self.graph.get_neighbors(node.id)))
            
            # Low confidence + some importance = worth investigating
            if node.confidence < 0.4 and connections >= 2:
                importance = self._calculate_importance(node)
                priority = GOAL_TYPE_WEIGHTS[GoalType.UNCERTAINTY] * (0.5 + importance * 0.5)
                
                goals.append(CuriosityGoal(
                    type=GoalType.UNCERTAINTY,
                    target=node.name,
                    question=f"What exactly is '{node.name}'?",
                    priority=priority,
                    context={
                        "current_confidence": node.confidence,
                        "connections": connections,
                    },
                    related_nodes=[node.id],
                ))
        
        # Sort by priority and limit
        goals.sort(key=lambda g: g.priority, reverse=True)
        return goals[:5]
    
    def _find_depth_opportunities(self) -> list[CuriosityGoal]:
        """Find important but shallow concepts."""
        goals = []
        
        shallow = self.analyzer.find_shallow_concepts(min_connections=4, max_properties=2)
        
        for node in shallow[:5]:
            connections = len(list(self.graph.get_neighbors(node.id)))
            properties = len(node.properties)
            
            priority = GOAL_TYPE_WEIGHTS[GoalType.DEPTH] * min(1.0, connections / 10)
            
            goals.append(CuriosityGoal(
                type=GoalType.DEPTH,
                target=node.name,
                question=f"What are the key properties and details of '{node.name}'?",
                priority=priority,
                context={
                    "connections": connections,
                    "current_properties": properties,
                    "reason": "Highly referenced but shallow representation",
                },
                related_nodes=[node.id],
            ))
        
        return goals
    
    def _find_verification_targets(self) -> list[CuriosityGoal]:
        """Find structurally anomalous claims that need verification."""
        goals = []
        
        anomalies = self.analyzer.find_anomalies(threshold=0.4, limit=5)
        
        for anomaly in anomalies:
            node = self.graph.get_node(anomaly.node_id)
            if not node:
                continue
            
            priority = GOAL_TYPE_WEIGHTS[GoalType.VERIFICATION] * anomaly.score
            
            # Build verification question based on what we know
            if node.properties.get("definition"):
                question = f"Is the definition of '{node.name}' correct: {node.properties['definition'][:50]}...?"
            else:
                question = f"Is '{node.name}' accurately represented in this knowledge base?"
            
            goals.append(CuriosityGoal(
                type=GoalType.VERIFICATION,
                target=node.name,
                question=question,
                priority=priority,
                context={
                    "anomaly_score": anomaly.score,
                    "reasons": anomaly.reasons,
                    "current_confidence": node.confidence,
                },
                related_nodes=[node.id],
            ))
        
        return goals
    
    def summarize(self) -> str:
        """Get a summary of current curiosity state."""
        goals = self.generate_goals(limit=20)
        
        if not goals:
            return "I'm not particularly curious about anything right now."
        
        # Group by type
        by_type = defaultdict(list)
        for goal in goals:
            by_type[goal.type].append(goal)
        
        lines = ["What I'm curious about:\n"]
        
        type_names = {
            GoalType.GAP: "Knowledge Gaps",
            GoalType.CONNECTION: "Potential Connections",
            GoalType.UNCERTAINTY: "Uncertainties",
            GoalType.DEPTH: "Shallow Concepts",
            GoalType.VERIFICATION: "Claims to Verify",
            GoalType.NOVELTY: "New Areas to Explore",
        }
        
        for goal_type, type_goals in by_type.items():
            lines.append(f"\n{type_names.get(goal_type, goal_type.value)}:")
            for goal in type_goals[:3]:
                lines.append(f"  - {goal.question} (priority: {goal.priority:.2f})")
        
        return "\n".join(lines)
