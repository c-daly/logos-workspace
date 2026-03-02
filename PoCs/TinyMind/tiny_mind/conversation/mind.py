"""
TinyMind - A baby intelligence that learns through conversation.

This is the main interface. It maintains a knowledge graph that
grows through every interaction.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from ..substrate.graph import KnowledgeGraph
from ..substrate.node import Node, TemporalGrain
from ..substrate.edge import Edge, Relations
from ..substrate.source import Source, SourceType
from ..extraction.extractor import Extractor, ExtractionResult, CritiqueResult
from ..revision.reviser import Reviser, RevisionResult
from ..curiosity.drive import CuriosityDrive
from ..curiosity.investigator import Investigator
from ..curiosity.goals import CuriosityGoal, GoalType, InvestigationResult
from .clustering import detect_communities, generate_cluster_colors, get_confidence_border


@dataclass
class ConversationTurn:
    """A single turn in conversation."""
    role: str  # "user" or "mind"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extracted: Optional[ExtractionResult] = None
    nodes_created: list[str] = field(default_factory=list)
    edges_created: list[str] = field(default_factory=list)


@dataclass
class SignificanceScore:
    """Multi-dimensional significance assessment."""
    surprise: float = 0.0          # Unexpected information
    contradiction: float = 0.0     # Conflicts with beliefs
    novelty: float = 0.0           # First encounter
    user_intensity: float = 0.0    # Emphatic input
    question_prompted: float = 0.0 # User asked something
    correction: float = 0.0        # User corrected us

    def total(self) -> float:
        """Weighted total significance."""
        weights = {
            'surprise': 2.0,
            'contradiction': 2.5,
            'novelty': 1.5,
            'user_intensity': 1.5,
            'question_prompted': 1.0,
            'correction': 3.0,  # Corrections are very significant
        }
        return sum(getattr(self, k) * v for k, v in weights.items())


class TinyMind:
    """
    A tiny baby intelligence that learns through conversation.

    Start it, talk to it, teach it things. It will build up
    a knowledge graph from scratch, learning whatever you share.
    """

    def __init__(
        self,
        name: str = "Tiny",
        llm_provider: str = "openai",
        model: str = None,
        save_path: str = None,
        # Critic uses a different model to catch extraction errors
        critic_provider: str = None,
        critic_model: str = None,
        use_critic: bool = True,
    ):
        self.name = name
        self.save_path = save_path or f"./{name.lower()}_mind.json"
        self.use_critic = use_critic

        # Core components
        self.graph = KnowledgeGraph()
        self.extractor = Extractor(
            llm_provider=llm_provider,
            model=model,
            critic_provider=critic_provider,
            critic_model=critic_model,
        )
        self.reviser = Reviser(
            llm_provider=llm_provider,
            model=model,
            log_path=f"./{name.lower()}_revision_log.json",
        )
        self.curiosity = CuriosityDrive(self.graph)
        self.investigator = Investigator(
            extractor=self.extractor,
            llm_provider=llm_provider,
            model=model,
        )

        # Conversation state
        self.history: list[ConversationTurn] = []
        self.recent_topics: list[str] = []

        # Learning statistics
        self.total_nodes_created = 0
        self.total_edges_created = 0
        self.session_start = datetime.now(timezone.utc)

        # Critique state (initialized to avoid AttributeError if methods called out of order)
        self._last_rejected: list[dict] = []
        self._last_critique_notes: str = ""

        # Try to load existing mind
        if os.path.exists(self.save_path):
            self.load()
        
        # Ensure core nodes (self, user) exist
        self._ensure_core_nodes()

    def _ensure_core_nodes(self):
        """
        Ensure self and user nodes exist in the graph.
        
        Called after initialization and loading to guarantee these
        foundational nodes are always present for introspection.
        """
        # Ensure self node exists
        if not self.graph.get_node("self"):
            self_node = Node(
                id="self",
                name=self.name,
                source=Source.bootstrap(f"{self.name} self-model"),
                temporal_grain=TemporalGrain.ETERNAL,
                confidence=1.0,
                properties={
                    "is_agent": True,
                    "is_self": True,
                    "description": "The mind's model of itself",
                },
            )
            self.graph.add_node(self_node)
        
        # Ensure user node exists
        if not self.graph.get_node("user"):
            user_node = Node(
                id="user",
                name="User",
                source=Source.bootstrap("Conversation partner model"),
                temporal_grain=TemporalGrain.ETERNAL,
                confidence=1.0,
                properties={
                    "is_agent": True,
                    "is_user": True,
                    "description": "The mind's model of its conversation partner",
                },
            )
            self.graph.add_node(user_node)

    def greet(self) -> str:
        """Generate a greeting based on current state."""
        stats = self.graph.get_stats()

        if stats.node_count <= 2:  # Just Self and Thing
            return (
                f"Hello! I'm {self.name}. I'm a tiny mind that learns through conversation.\n\n"
                f"I don't know much yet - just that I exist and can learn.\n"
                f"Please teach me something!"
            )
        else:
            orphans = self.graph.get_orphans()
            curious_about = [n.name for n in orphans[:3]] if orphans else []

            greeting = (
                f"Hello! I'm {self.name}. I know about {stats.node_count} things "
                f"with {stats.edge_count} connections between them.\n"
            )

            if curious_about:
                greeting += f"\nI'm curious about: {', '.join(curious_about)} - they seem disconnected."

            return greeting

    def hear(self, text: str) -> str:
        """
        Process input from the user.

        This is the main learning entry point. The mind will:
        1. Check if this is a question to answer
        2. Extract knowledge from the input
        3. Critique the extraction (if enabled)
        4. Integrate approved knowledge into the graph
        5. Respond based on what it learned
        """
        # Check if this is a question we should answer
        if self._is_question(text):
            answer = self._answer_question(text)
            if answer:
                self.history.append(ConversationTurn(role="user", content=text))
                self.history.append(ConversationTurn(role="mind", content=answer))
                return answer

        # Record the turn
        turn = ConversationTurn(role="user", content=text)

        # Extract knowledge first (to avoid duplicate API calls)
        result = self.extractor.extract(text, self.graph)
        turn.extracted = result

        # Assess significance (pass extraction result to avoid duplicate extraction)
        significance = self._assess_significance(text, extraction_result=result)

        rejected = []
        if self.use_critic and (result.relations or result.entities):
            # Run critique pass with (potentially different) model
            critique = self.extractor.critique(result, text)

            # Integrate only approved/corrected knowledge
            nodes, edges, rejected = self.extractor.integrate_critiqued(
                critique, self.graph, text
            )

            # Track what was rejected for the response
            self._last_rejected = rejected
            self._last_critique_notes = critique.critique_notes
        else:
            # No critic - integrate everything (original behavior)
            nodes, edges = self.extractor.integrate(result, self.graph, text)
            self._last_rejected = []
            self._last_critique_notes = ""

        turn.nodes_created = [n.id for n in nodes]
        turn.edges_created = [e.id for e in edges]

        # Update stats
        self.total_nodes_created += len(nodes)
        self.total_edges_created += len(edges)

        # Track topics
        for entity in result.entities:
            if entity.name not in self.recent_topics:
                self.recent_topics.append(entity.name)
        self.recent_topics = self.recent_topics[-20:]  # Keep last 20

        self.history.append(turn)

        # Introspect - generate mental content about this interaction
        mental_nodes, mental_edges = self._introspect(
            user_input=text,
            extraction_result=result,
            nodes_created=nodes,
            edges_created=edges,
        )

        # Generate response (now informed by introspection)
        response = self._generate_response(
            result, nodes, edges, significance,
            mental_nodes=mental_nodes, mental_edges=mental_edges
        )

        # Record mind's response
        self.history.append(ConversationTurn(role="mind", content=response))

        # Auto-save periodically
        if len(self.history) % 10 == 0:
            self.save()

        return response

    def _assess_significance(self, text: str, extraction_result: ExtractionResult = None) -> SignificanceScore:
        """Assess how significant this input is.
        
        Args:
            text: The input text
            extraction_result: Optional pre-computed extraction result to avoid duplicate API calls
        """
        score = SignificanceScore()

        # Simple heuristics for now - could be LLM-enhanced later

        # Check for question marks (user asking)
        if "?" in text:
            score.question_prompted = 0.5

        # Check for correction signals
        correction_signals = ["no,", "actually", "wrong", "not quite", "incorrect", "that's not"]
        if any(signal in text.lower() for signal in correction_signals):
            score.correction = 0.8

        # Check for emphasis (caps, exclamation)
        if text.isupper() or text.count("!") > 1:
            score.user_intensity = 0.6

        # Novelty - how many new entities? (only if extraction result provided)
        if extraction_result is not None:
            new_entities = [e for e in extraction_result.entities if not self.graph.find_node(e.name)]
            if new_entities:
                score.novelty = min(len(new_entities) * 0.2, 1.0)

        return score

    def _generate_response(
        self,
        result: ExtractionResult,
        nodes: list[Node],
        edges: list[Edge],
        significance: SignificanceScore,
        mental_nodes: list[Node] = None,
        mental_edges: list[Edge] = None,
    ) -> str:
        """Generate a response based on what was learned and introspection."""
        parts = []
        mental_nodes = mental_nodes or []
        mental_edges = mental_edges or []

        # Gather introspective content by type
        thoughts = [n for n in mental_nodes if n.properties.get("mental_type") == "thought"]
        perceptions = [n for n in mental_nodes if n.properties.get("mental_type") == "perception"]
        feelings = [n for n in mental_nodes if n.properties.get("mental_type") == "feeling"]

        # Lead with a thought if we have one (more natural than "I learned...")
        if thoughts:
            # Prioritize realizations and connections over confusion
            realization = next(
                (t for t in thoughts if t.properties.get("thought_type") == "realization"), 
                None
            )
            connection = next(
                (t for t in thoughts if t.properties.get("thought_type") == "connection"), 
                None
            )
            curiosity = next(
                (t for t in thoughts if t.properties.get("thought_type") == "curiosity"),
                None
            )
            
            if realization:
                parts.append(f"Oh! {realization.name}")
            elif connection:
                parts.append(connection.name)
            elif curiosity:
                parts.append(f"Hmm, {curiosity.name}")

        # Acknowledge new learning (but more naturally)
        if nodes:
            node_names = [n.name for n in nodes[:5]]
            if len(nodes) > 5:
                node_names.append(f"and {len(nodes) - 5} more")
            
            if not thoughts:  # Only use template if no thoughts to express
                parts.append(f"I learned about: {', '.join(node_names)}")
            else:
                # Just note we learned something new
                parts.append(f"(Learned: {', '.join(node_names[:3])})")

        if edges and not thoughts:
            # Summarize a few key relationships
            sample_edges = edges[:3]
            for edge in sample_edges:
                source = self.graph.get_node(edge.source_id)
                target = self.graph.get_node(edge.target_id)
                if source and target:
                    parts.append(f"  - {source.name} {edge.relation} {target.name}")

        # Show rejected relations (critic caught something!)
        rejected = getattr(self, '_last_rejected', [])
        if rejected:
            parts.append(f"\n[Critic rejected {len(rejected)} claim(s)]")
            for rej in rejected[:2]:  # Show first 2
                rel = rej.get('relation')
                reason = rej.get('reason', 'no reason')
                if rel:
                    parts.append(f"  - '{rel.source} {rel.relation} {rel.target}': {reason}")

        # Express confusion if present
        confusion = next(
            (t for t in thoughts if t.properties.get("thought_type") == "confusion"),
            None
        )
        if confusion:
            parts.append(f"\n{confusion.name}")

        # Note uncertainties from extraction
        if result.uncertainties and not confusion:
            parts.append(f"\nI'm uncertain about: {result.uncertainties[0].description}")

        # Express curiosity naturally
        curiosity_thought = next(
            (t for t in thoughts if t.properties.get("thought_type") == "curiosity"),
            None
        )
        if curiosity_thought and curiosity_thought.name not in str(parts):
            parts.append(f"\n{curiosity_thought.name}")
        elif len(self.history) % 5 == 0:
            # Fallback to orphan curiosity
            orphans = self.graph.get_orphans()
            orphans = [o for o in orphans if o.id not in ("self", "thing", "user")]
            if orphans:
                parts.append(f"\nI'm curious how '{orphans[0].name}' connects to other things...")

        # Acknowledge perceived user state if notable
        user_perception = next(
            (p for p in perceptions if p.properties.get("about") == "user"),
            None
        )
        if user_perception and "frustrat" in user_perception.name.lower():
            parts.insert(0, "I sense I might have missed something important.")

        # If nothing was learned, acknowledge that
        if not nodes and not edges and not thoughts:
            parts.append("I'm not sure I learned anything new from that. Could you tell me more?")

        # Show current state briefly
        stats = self.graph.get_stats()
        parts.append(f"\n[Knowledge: {stats.node_count} nodes, {stats.edge_count} edges]")

        return "\n".join(parts)


    def _introspect(
        self,
        user_input: str,
        extraction_result: ExtractionResult,
        nodes_created: list[Node],
        edges_created: list[Edge],
    ) -> tuple[list[Node], list[Edge]]:
        """
        Generate internal mental content based on processing state.
        
        Creates nodes for thoughts, perceptions, and feelings, connecting
        them to the self node and relevant concepts.
        
        Returns:
            Tuple of (mental_nodes, mental_edges) created
        """
        import uuid
        
        mental_nodes = []
        mental_edges = []
        
        # Build context for introspection
        extraction_summary = ""
        if extraction_result.entities:
            entity_names = [e.name for e in extraction_result.entities[:5]]
            extraction_summary += f"Entities: {', '.join(entity_names)}\n"
        if extraction_result.relations:
            rel_strs = [f"{r.source} {r.relation} {r.target}" for r in extraction_result.relations[:3]]
            extraction_summary += f"Relations: {'; '.join(rel_strs)}\n"
        if not extraction_summary:
            extraction_summary = "Nothing new extracted"
        
        # Get relevant existing knowledge
        existing_context = ""
        if self.recent_topics:
            existing_context = f"Recent topics: {', '.join(self.recent_topics[-5:])}"
        
        # Format recent conversation history
        history_lines = []
        for turn in self.history[-4:]:
            prefix = "User:" if turn.role == "user" else "Mind:"
            history_lines.append(f"{prefix} {turn.content[:100]}...")
        history = "\n".join(history_lines) if history_lines else "No prior conversation"
        
        # Call LLM for introspection
        introspection = self.extractor.introspect(
            user_input=user_input,
            extraction_summary=extraction_summary,
            existing_context=existing_context,
            history=history,
        )
        
        # Handle errors gracefully
        if introspection.get("error"):
            return [], []
        
        self_node = self.graph.get_node("self")
        user_node = self.graph.get_node("user")
        
        # Process perceptions
        for perception in introspection.get("perceptions", []):
            node_id = f"perception_{uuid.uuid4().hex[:8]}"
            node = Node(
                id=node_id,
                name=perception["content"],
                source=Source(
                    source_type=SourceType.SELF_REFLECTION,
                    raw_content=user_input,
                ),
                temporal_grain=TemporalGrain(perception.get("temporal_grain", "minutes")),
                confidence=perception.get("confidence", 0.5),
                properties={
                    "mental_type": "perception",
                    "about": perception.get("about", "user"),
                },
            )
            self.graph.add_node(node)
            mental_nodes.append(node)
            
            # Connect self --perceives--> perception
            edge = Edge(
                source_id="self",
                target_id=node_id,
                relation="perceives",
                source=Source(source_type=SourceType.SELF_REFLECTION),
                confidence=perception.get("confidence", 0.5),
            )
            self.graph.add_edge(edge)
            mental_edges.append(edge)
            
            # If about user, also connect perception --about--> user
            if perception.get("about") == "user" and user_node:
                about_edge = Edge(
                    source_id=node_id,
                    target_id="user",
                    relation="about",
                    source=Source(source_type=SourceType.SELF_REFLECTION),
                )
                self.graph.add_edge(about_edge)
                mental_edges.append(about_edge)
        
        # Process thoughts
        for thought in introspection.get("thoughts", []):
            node_id = f"thought_{uuid.uuid4().hex[:8]}"
            node = Node(
                id=node_id,
                name=thought["content"],
                source=Source(
                    source_type=SourceType.SELF_REFLECTION,
                    raw_content=user_input,
                ),
                temporal_grain=TemporalGrain(thought.get("temporal_grain", "minutes")),
                confidence=thought.get("confidence", 0.5),
                properties={
                    "mental_type": "thought",
                    "thought_type": thought.get("thought_type", "connection"),
                    "about_concepts": thought.get("about", []),
                },
            )
            self.graph.add_node(node)
            mental_nodes.append(node)
            
            # Connect self --has_thought--> thought
            edge = Edge(
                source_id="self",
                target_id=node_id,
                relation="has_thought",
                source=Source(source_type=SourceType.SELF_REFLECTION),
                confidence=thought.get("confidence", 0.5),
            )
            self.graph.add_edge(edge)
            mental_edges.append(edge)
            
            # Connect thought --about--> each concept it references
            for concept_name in thought.get("about", []):
                concept_node = self.graph.find_node(concept_name)
                if concept_node:
                    about_edge = Edge(
                        source_id=node_id,
                        target_id=concept_node.id,
                        relation="about",
                        source=Source(source_type=SourceType.SELF_REFLECTION),
                    )
                    self.graph.add_edge(about_edge)
                    mental_edges.append(about_edge)
        
        # Process feelings
        for feeling in introspection.get("feelings", []):
            node_id = f"feeling_{uuid.uuid4().hex[:8]}"
            node = Node(
                id=node_id,
                name=feeling["content"],
                source=Source(
                    source_type=SourceType.SELF_REFLECTION,
                    raw_content=user_input,
                ),
                temporal_grain=TemporalGrain(feeling.get("temporal_grain", "minutes")),
                confidence=feeling.get("intensity", 0.5),
                properties={
                    "mental_type": "feeling",
                    "valence": feeling.get("valence", "neutral"),
                    "intensity": feeling.get("intensity", 0.5),
                },
            )
            self.graph.add_node(node)
            mental_nodes.append(node)
            
            # Connect self --feels--> feeling
            edge = Edge(
                source_id="self",
                target_id=node_id,
                relation="feels",
                source=Source(source_type=SourceType.SELF_REFLECTION),
                confidence=feeling.get("intensity", 0.5),
            )
            self.graph.add_edge(edge)
            mental_edges.append(edge)
        
        return mental_nodes, mental_edges

    def _is_question(self, text: str) -> bool:
        """Detect if this is a question we should answer rather than learn from."""
        text_lower = text.lower().strip()

        # Must have question mark or start with question words
        has_question_mark = "?" in text
        question_starters = [
            "what ", "what's", "whats",
            "who ", "who's", "whos",
            "where ", "where's",
            "when ", "when's",
            "why ", "why's",
            "how ", "how's",
            "do you know", "does ", "did ",
            "can you tell", "could you tell",
            "tell me about", "explain ",
            "is ", "are ", "was ", "were ",
        ]

        starts_with_question = any(text_lower.startswith(q) for q in question_starters)

        return has_question_mark or starts_with_question

    def _answer_question(self, text: str) -> str:
        """Answer a question from the knowledge graph."""
        text_lower = text.lower()

        # Extract potential topics from the question
        # Remove common question words to find the subject
        for word in ["what", "who", "where", "when", "why", "how", "is", "are",
                     "do", "does", "did", "can", "could", "tell", "me", "about",
                     "explain", "you", "know", "the", "a", "an", "?"]:
            text_lower = text_lower.replace(word, " ")

        # Get candidate topics (words/phrases)
        words = [w.strip() for w in text_lower.split() if len(w.strip()) > 2]

        # Try to find matching nodes
        found_nodes = []
        for word in words:
            node = self.graph.find_node(word)
            if node:
                found_nodes.append(node)

        # Also try multi-word phrases
        if len(words) >= 2:
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                node = self.graph.find_node(phrase)
                if node and node not in found_nodes:
                    found_nodes.append(node)

        if not found_nodes:
            # No direct matches - try fuzzy search
            for word in words:
                matches = self.graph.search_nodes(word, limit=3)
                for node in matches:
                    if node not in found_nodes:
                        found_nodes.append(node)

        if not found_nodes:
            return None  # Fall through to normal extraction

        # Build answer from found nodes
        parts = []
        for node in found_nodes[:3]:  # Limit to 3 main topics
            parts.append(f"**{node.name}**")

            # Add definition if available
            if node.properties.get("definition"):
                parts.append(f"  Definition: {node.properties['definition']}")
            elif node.properties.get("is_a"):
                parts.append(f"  Is a: {node.properties['is_a']}")

            # Add key properties
            for key, value in list(node.properties.items())[:3]:
                if key not in ("definition", "is_a"):
                    parts.append(f"  {key}: {value}")

            # Add connections
            edges_out = self.graph.find_edges(source_id=node.id)
            edges_in = self.graph.find_edges(target_id=node.id)

            if edges_out or edges_in:
                parts.append("  Connections:")
                for edge in edges_out[:3]:
                    target = self.graph.get_node(edge.target_id)
                    if target:
                        parts.append(f"    → {edge.relation} {target.name}")
                for edge in edges_in[:3]:
                    source = self.graph.get_node(edge.source_id)
                    if source:
                        parts.append(f"    ← {source.name} {edge.relation}")

            parts.append("")  # Blank line between topics

        stats = self.graph.get_stats()
        parts.append(f"[Knowledge: {stats.node_count} nodes, {stats.edge_count} edges]")

        return "\n".join(parts)

    def ask(self, question: str) -> str:
        """
        The mind asks the user a question to learn more.
        """
        questions = self.extractor.generate_questions(self.graph, self.recent_topics)
        if questions:
            q = questions[0]
            return f"I'm wondering: {q['question']}\n(Because: {q['motivation']})"
        return "I'm not sure what to ask right now..."

    def reflect(self) -> str:
        """
        The mind reflects on its current knowledge.
        """
        analysis = self.extractor.reflect(self.graph)

        parts = ["Reflecting on what I know...\n"]

        if analysis.get("orphans"):
            parts.append("Disconnected concepts:")
            for orphan in analysis["orphans"][:3]:
                parts.append(f"  - {orphan.get('node_id')}: {orphan.get('suggestion', '?')}")

        if analysis.get("gaps"):
            parts.append("\nMissing connections:")
            for gap in analysis["gaps"][:3]:
                parts.append(f"  - {gap.get('from')} → ? → {gap.get('to')}")

        if analysis.get("contradictions"):
            parts.append("\nContradictions:")
            for c in analysis["contradictions"][:3]:
                parts.append(f"  - {c.get('description')}")

        return "\n".join(parts)

    def know(self, topic: str = None) -> str:
        """
        Report what the mind knows, optionally about a topic.
        """
        if topic:
            # Try exact match first
            node = self.graph.find_node(topic)

            # If no exact match, try search
            if not node:
                matches = self.graph.search_nodes(topic, limit=15)
                if not matches:
                    return f"I don't know about '{topic}' yet."

                # If only one match, use it
                if len(matches) == 1:
                    node = matches[0]
                else:
                    # Multiple matches - show list with full details
                    parts = [f"I know about {len(matches)} things related to '{topic}':\n"]
                    for m in matches[:15]:
                        conf_pct = int(m.confidence * 100)
                        type_info = m.properties.get('is_a', '')
                        if isinstance(type_info, list):
                            type_info = type_info[0] if type_info else ''
                        type_str = f" ({type_info})" if type_info else ""
                        
                        defn = m.properties.get('definition', '')
                        if defn:
                            # Show more of the definition
                            if len(str(defn)) > 120:
                                defn = f"\n      {defn[:120]}..."
                            else:
                                defn = f"\n      {defn}"
                        
                        parts.append(f"  • {m.name}{type_str} [{conf_pct}%]{defn}")
                    
                    if len(matches) > 15:
                        parts.append(f"\n  ... and {len(matches) - 15} more")
                    parts.append(f"\nUse /know <name> for details on a specific item.")
                    return "\n".join(parts)

            # Show detailed info about single node
            parts = [f"\n{'='*50}"]
            parts.append(f"  {node.name}")
            parts.append(f"{'='*50}")
            
            # Confidence with visual bar
            conf_pct = int(node.confidence * 100)
            bar_len = conf_pct // 5
            conf_bar = "█" * bar_len + "░" * (20 - bar_len)
            parts.append(f"\n  Confidence: [{conf_bar}] {conf_pct}%")
            
            # Type info
            type_info = node.properties.get('is_a')
            if type_info:
                if isinstance(type_info, list):
                    type_info = ", ".join(type_info)
                parts.append(f"  Type: {type_info}")
            
            # Temporal info
            grain = node.temporal_grain.value if hasattr(node.temporal_grain, 'value') else str(node.temporal_grain)
            parts.append(f"  Temporal: {grain}")
            
            # Definition (full, with word wrap)
            defn = node.properties.get('definition')
            if defn:
                parts.append(f"\n  Definition:")
                # Word wrap at ~70 chars
                words = str(defn).split()
                line = "    "
                for word in words:
                    if len(line) + len(word) > 74:
                        parts.append(line)
                        line = "    "
                    line += word + " "
                if line.strip():
                    parts.append(line)
            
            # Other properties (not already shown)
            skip_keys = {'is_a', 'definition', 'is_rule', 'condition', 'consequence', 'domain', 'auto_created', 'from_relation'}
            other_props = [(k, v) for k, v in node.properties.items() if k not in skip_keys]
            
            if other_props:
                parts.append(f"\n  Properties:")
                for k, v in other_props[:10]:
                    v_str = str(v)
                    if len(v_str) > 80:
                        v_str = v_str[:80] + "..."
                    parts.append(f"    {k}: {v_str}")
                if len(other_props) > 10:
                    parts.append(f"    ... and {len(other_props) - 10} more")
            
            # Rule details
            if node.properties.get('is_rule'):
                parts.append(f"\n  Rule:")
                parts.append(f"    IF: {node.properties.get('condition', '?')}")
                parts.append(f"    THEN: {node.properties.get('consequence', '?')}")
                if node.properties.get('domain'):
                    parts.append(f"    Domain: {node.properties.get('domain')}")

            # Connections (show more)
            out_edges = self.graph.find_edges(source_id=node.id)
            in_edges = self.graph.find_edges(target_id=node.id)
            
            if out_edges or in_edges:
                parts.append(f"\n  Connections ({len(out_edges) + len(in_edges)} total):")
                
                if out_edges:
                    parts.append(f"    Outgoing:")
                    for edge in out_edges[:8]:
                        target = self.graph.get_node(edge.target_id)
                        if target:
                            conf = int(edge.confidence * 100)
                            parts.append(f"      → {edge.relation} → {target.name} [{conf}%]")
                    if len(out_edges) > 8:
                        parts.append(f"      ... and {len(out_edges) - 8} more")

                if in_edges:
                    parts.append(f"    Incoming:")
                    for edge in in_edges[:8]:
                        source = self.graph.get_node(edge.source_id)
                        if source:
                            conf = int(edge.confidence * 100)
                            parts.append(f"      {source.name} → {edge.relation} → [{conf}%]")
                    if len(in_edges) > 8:
                        parts.append(f"      ... and {len(in_edges) - 8} more")

            return "\n".join(parts)
        else:
            # General knowledge summary
            stats = self.graph.get_stats()
            top_nodes = self.graph.get_highly_connected(15)

            parts = [
                f"\n{'='*50}",
                f"  Knowledge Summary",
                f"{'='*50}",
                f"\n  Total: {stats.node_count} concepts, {stats.edge_count} connections",
                f"  Average confidence: {stats.avg_confidence:.0%}",
                f"  Orphan nodes: {stats.orphan_count}",
            ]
            
            # Relation types
            rel_counts = stats.relation_types
            if rel_counts:
                parts.append(f"\n  Relation types:")
                for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1])[:8]:
                    parts.append(f"    {rel}: {count}")
            
            # Top connected nodes
            if top_nodes:
                parts.append(f"\n  Most connected concepts:")
                for node, count in top_nodes[:10]:
                    conf_pct = int(node.confidence * 100)
                    type_info = node.properties.get('is_a', '')
                    if isinstance(type_info, list):
                        type_info = type_info[0] if type_info else ''
                    type_str = f" ({type_info})" if type_info else ""
                    parts.append(f"    • {node.name}{type_str} - {count} connections [{conf_pct}%]")

            return "\n".join(parts)

    def forget(self, topic: str) -> str:
        """
        Remove a concept from the mind.
        """
        node = self.graph.find_node(topic)
        if not node:
            return f"I don't know about '{topic}'."

        if node.id in ("self", "thing"):
            return "I can't forget that - it's fundamental to my existence."

        self.graph.remove_node(node.id)
        return f"I've forgotten about '{topic}'."

    def save(self):
        """Save the mind state to disk."""
        data = {
            "name": self.name,
            "graph": self.graph.to_dict(),
            "history_length": len(self.history),
            "recent_topics": self.recent_topics,
            "stats": {
                "total_nodes_created": self.total_nodes_created,
                "total_edges_created": self.total_edges_created,
                "session_start": self.session_start.isoformat(),
            },
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load the mind state from disk."""
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)

            self.name = data.get("name", self.name)
            self.graph = KnowledgeGraph.from_dict(data.get("graph", {}))
            self.recent_topics = data.get("recent_topics", [])
            self.total_nodes_created = data.get("stats", {}).get("total_nodes_created", 0)
            self.total_edges_created = data.get("stats", {}).get("total_edges_created", 0)

            # Update references in components that hold graph reference
            self.curiosity = CuriosityDrive(self.graph)
            self.investigator.extractor = self.extractor
            
            # Ensure core nodes exist (for backwards compatibility with old saves)
            self._ensure_core_nodes()

        except Exception as e:
            print(f"Could not load mind state: {e}")

    def revise(self, verbose: bool = True) -> RevisionResult:
        """
        Run maintenance pass on knowledge graph.
        
        Performs:
        - Deduplication: Merge semantically similar nodes
        - Contradiction resolution: Handle conflicting relations
        - Pruning: Remove low-confidence orphan nodes
        
        All actions are auto-accepted and logged.
        """
        result = self.reviser.revise(self.graph, verbose=verbose)
        return result

    def wonder(self, limit: int = 5) -> list[CuriosityGoal]:
        """
        What am I curious about?
        
        Analyzes the knowledge graph to identify:
        - Knowledge gaps (missing definitions)
        - Potential connections
        - Uncertain claims
        - Shallow concepts
        - Anomalies to verify
        
        Returns prioritized list of curiosity goals.
        """
        return self.curiosity.generate_goals(limit=limit)
    
    def explore(self, goal: CuriosityGoal = None, verbose: bool = True) -> InvestigationResult:
        """
        Pursue a curiosity goal using web search and other tools.
        
        If no goal provided, investigates the top curiosity goal.
        
        Returns InvestigationResult with what was learned.
        """
        if goal is None:
            goals = self.wonder(limit=1)
            if not goals:
                return InvestigationResult(
                    goal=CuriosityGoal(
                        type=None,
                        target="",
                        question="Nothing to investigate",
                        priority=0,
                    ),
                    success=False,
                    errors=["No curiosity goals found"],
                )
            goal = goals[0]
        
        result = self.investigator.investigate(goal, self.graph, verbose=verbose)
        
        # Mark as investigated to avoid re-investigating soon
        self.curiosity.mark_investigated(goal.target)
        
        return result
    
    def ponder(self, cycles: int = 3, verbose: bool = True) -> list[InvestigationResult]:
        """
        Autonomous curiosity-driven learning.
        
        Runs multiple exploration cycles, investigating top curiosity goals.
        
        Returns list of investigation results.
        """
        results = []
        
        if verbose:
            print("=" * 50)
            print("  PONDERING - Autonomous Exploration")
            print("=" * 50)
        
        for i in range(cycles):
            if verbose:
                print(f"\n--- Cycle {i + 1}/{cycles} ---")
            
            result = self.explore(verbose=verbose)
            results.append(result)
            
            if not result.success:
                if verbose:
                    print(f"  Stopping: {result.errors}")
                break
        
        if verbose:
            total_nodes = sum(r.nodes_added for r in results)
            total_edges = sum(r.edges_added for r in results)
            print(f"\n{'=' * 50}")
            print(f"Pondering complete: {len(results)} investigations")
            print(f"Total learned: +{total_nodes} nodes, +{total_edges} edges")
            print("=" * 50)
        
        return results

    def research(self, topic: str, verbose: bool = True) -> InvestigationResult:
        """
        Research a topic using web search.

        Unlike explore() which follows the mind's curiosity,
        research() investigates what the user asks for.

        Args:
            topic: What to research (e.g., "quantum computing", "French Revolution")
            verbose: Print progress

        Returns:
            InvestigationResult with what was learned.
        """
        # Create a user-directed goal
        goal = CuriosityGoal(
            type=GoalType.NOVELTY,
            target=topic,
            question=f"What is {topic}?",
            priority=1.0,  # User-directed = high priority
            context={"source": "user_request"},
        )

        if verbose:
            print(f"\nResearching: {topic}")

        result = self.investigator.investigate(goal, self.graph, verbose=verbose)

        # Mark as investigated
        self.curiosity.mark_investigated(topic)

        return result

    def read_pdf(
        self,
        filepath: str,
        max_chars_per_chunk: int = 2000,
        verbose: bool = True,
        start_page: int = 0,
        end_page: int = None,
        parallel: bool = False,
        max_workers: int = 4,
        batch_size: int = 10,
    ) -> dict:
        """
        Read a PDF and extract knowledge from it.

        Args:
            filepath: Path to the PDF file
            max_chars_per_chunk: Maximum characters per chunk to process
            verbose: Print progress
            start_page: First page to read (0-indexed)
            end_page: Last page to read (exclusive), None for all
            parallel: Use parallel processing for extraction
            max_workers: Number of parallel workers (if parallel=True)
            batch_size: Pages per batch for hybrid mode (parallel=True)

        Returns:
            Summary dict with stats about what was learned
        """
        try:
            import pymupdf as fitz  # PyMuPDF >= 1.24
        except ImportError:
            try:
                import fitz  # PyMuPDF < 1.24
            except ImportError:
                raise ImportError("PyMuPDF required. pip install pymupdf")

        if verbose:
            print(f"Reading PDF: {filepath}")

        doc = fitz.open(filepath)
        total_pages = len(doc)
        end_page = end_page or total_pages
        start_page = max(0, start_page)
        end_page = min(end_page, total_pages)

        # Collect all chunks with page info
        all_chunks = []
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            text = page.get_text()

            if not text.strip():
                continue

            chunks = self._chunk_text(text, max_chars_per_chunk)
            for chunk in chunks:
                if len(chunk.strip()) >= 50:
                    all_chunks.append((page_num, chunk))

        doc.close()

        if verbose:
            print(f"  {len(all_chunks)} chunks from pages {start_page + 1}-{end_page}")

        if not all_chunks:
            return {"pages": total_pages, "chunks_processed": 0, "nodes_created": 0, "edges_created": 0}

        if parallel and len(all_chunks) > 1:
            return self._read_pdf_hybrid(all_chunks, total_pages, verbose, max_workers, batch_size)
        else:
            return self._read_pdf_sequential(all_chunks, total_pages, verbose)

    def _read_pdf_sequential(self, chunks: list, total_pages: int, verbose: bool) -> dict:
        """Process chunks sequentially."""
        total_nodes = 0
        total_edges = 0
        chunks_processed = 0

        for page_num, chunk in chunks:
            if verbose:
                print(f"  Page {page_num + 1}/{total_pages}, chunk {chunks_processed + 1}...")

            before_nodes = len(list(self.graph.nodes()))
            before_edges = len(list(self.graph.edges()))

            try:
                self.hear(chunk)
            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")
                continue

            new_nodes = len(list(self.graph.nodes())) - before_nodes
            new_edges = len(list(self.graph.edges())) - before_edges
            total_nodes += new_nodes
            total_edges += new_edges
            chunks_processed += 1

            if verbose and (new_nodes > 0 or new_edges > 0):
                print(f"    +{new_nodes} nodes, +{new_edges} edges")

        if verbose:
            print(f"\nDone! Processed {chunks_processed} chunks")
            print(f"Total learned: {total_nodes} nodes, {total_edges} edges")

        return {
            "pages": total_pages,
            "chunks_processed": chunks_processed,
            "nodes_created": total_nodes,
            "edges_created": total_edges,
        }

    def _read_pdf_hybrid(self, chunks: list, total_pages: int, verbose: bool, max_workers: int, batch_size: int) -> dict:
        """
        Hybrid parallel processing: batches processed sequentially, chunks within batch in parallel.

        This allows knowledge to accumulate between batches while still getting parallel speedup.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Group chunks into batches by page ranges
        batches = []
        current_batch = []
        current_batch_start_page = None

        for page_num, chunk in chunks:
            if current_batch_start_page is None:
                current_batch_start_page = page_num

            # Start new batch if we've exceeded batch_size pages
            if page_num >= current_batch_start_page + batch_size and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_start_page = page_num

            current_batch.append((page_num, chunk))

        if current_batch:
            batches.append(current_batch)

        if verbose:
            print(f"  {len(batches)} batches, {max_workers} workers per batch")

        total_nodes = 0
        total_edges = 0
        chunks_processed = 0

        for batch_idx, batch in enumerate(batches):
            batch_start = batch[0][0] + 1
            batch_end = batch[-1][0] + 1

            if verbose:
                print(f"\n  Batch {batch_idx + 1}/{len(batches)} (pages {batch_start}-{batch_end}, {len(batch)} chunks)")

            # Phase 1: Extract all chunks in this batch in parallel
            extraction_results = []

            def extract_chunk(page_chunk):
                page_num, chunk = page_chunk
                try:
                    # Each extraction sees current graph state (accumulated from previous batches)
                    result = self.extractor.extract(chunk, self.graph)
                    critique = self.extractor.critique(result, chunk) if self.use_critic else None
                    return (page_num, chunk, result, critique, None)
                except Exception as e:
                    return (page_num, chunk, None, None, str(e))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(extract_chunk, pc): pc for pc in batch}

                for future in as_completed(futures):
                    result = future.result()
                    extraction_results.append(result)
                    if verbose:
                        err = result[4]
                        if err:
                            print(f"    Error: {err[:50]}")

            # Phase 2: Integrate this batch's results sequentially
            batch_nodes = 0
            batch_edges = 0

            for page_num, chunk, result, critique, error in extraction_results:
                if error or not result:
                    continue

                before_nodes = len(list(self.graph.nodes()))
                before_edges = len(list(self.graph.edges()))

                try:
                    if critique and self.use_critic:
                        self.extractor.integrate_critiqued(critique, self.graph, chunk)
                    else:
                        self.extractor.integrate(result, self.graph, chunk)
                except Exception as e:
                    continue

                batch_nodes += len(list(self.graph.nodes())) - before_nodes
                batch_edges += len(list(self.graph.edges())) - before_edges
                chunks_processed += 1

            total_nodes += batch_nodes
            total_edges += batch_edges

            if verbose:
                print(f"    +{batch_nodes} nodes, +{batch_edges} edges (total: {len(list(self.graph.nodes()))} nodes)")

        if verbose:
            print(f"\nDone! Processed {chunks_processed} chunks in {len(batches)} batches")
            print(f"Total learned: {total_nodes} nodes, {total_edges} edges")

        return {
            "pages": total_pages,
            "chunks_processed": chunks_processed,
            "nodes_created": total_nodes,
            "edges_created": total_edges,
            "batches": len(batches),
        }

    def _chunk_text(self, text: str, max_chars: int) -> list[str]:
        """Split text into chunks, trying to break at paragraph boundaries."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If single paragraph is too long, split by sentences
                if len(para) > max_chars:
                    sentences = para.replace('. ', '.\n').split('\n')
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= max_chars:
                            current_chunk += sent + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sent + " "
                else:
                    current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def visualize(self, limit: int = 100) -> str:
        """Generate a text visualization of the graph."""
        lines = []
        lines.append("=" * 60)
        lines.append("  KNOWLEDGE GRAPH VISUALIZATION")
        lines.append("=" * 60)

        # Get stats
        stats = self.graph.get_stats()
        lines.append(f"\n  {stats.node_count} nodes, {stats.edge_count} edges")
        lines.append(f"  Average confidence: {stats.avg_confidence:.0%}")

        # Sort nodes by connection count (most connected first)
        nodes_with_counts = []
        for node in self.graph.nodes():
            out_edges = self.graph.find_edges(source_id=node.id)
            in_edges = self.graph.find_edges(target_id=node.id)
            conn_count = len(out_edges) + len(in_edges)
            nodes_with_counts.append((node, out_edges, in_edges, conn_count))

        nodes_with_counts.sort(key=lambda x: -x[3])

        # Show top connected nodes with their relationships
        lines.append("\n" + "-" * 60)
        lines.append("  TOP CONNECTED CONCEPTS")
        lines.append("-" * 60)
        
        shown = 0
        for node, out_edges, in_edges, conn_count in nodes_with_counts[:20]:
            if conn_count == 0:
                continue
            if shown >= limit:
                break

            # Node header with confidence bar
            conf_pct = int(node.confidence * 100)
            bar = "█" * (conf_pct // 10) + "░" * (10 - conf_pct // 10)
            type_info = node.properties.get('is_a', '')
            if isinstance(type_info, list):
                type_info = type_info[0] if type_info else ''
            type_str = f" ({type_info})" if type_info else ""
            
            lines.append(f"\n  ● {node.name}{type_str}")
            lines.append(f"    Confidence: [{bar}] {conf_pct}%  |  Connections: {conn_count}")
            
            # Show definition if present
            defn = node.properties.get('definition')
            if defn:
                defn_str = str(defn)[:100] + "..." if len(str(defn)) > 100 else str(defn)
                lines.append(f"    Definition: {defn_str}")

            # Outgoing relationships (show more)
            if out_edges:
                lines.append(f"    Outgoing ({len(out_edges)}):")
                for edge in out_edges[:5]:
                    target = self.graph.get_node(edge.target_id)
                    if target:
                        edge_conf = int(edge.confidence * 100)
                        lines.append(f"      → {edge.relation} → {target.name} [{edge_conf}%]")
                if len(out_edges) > 5:
                    lines.append(f"      ... and {len(out_edges) - 5} more")
            
            # Incoming relationships
            if in_edges:
                lines.append(f"    Incoming ({len(in_edges)}):")
                for edge in in_edges[:5]:
                    source = self.graph.get_node(edge.source_id)
                    if source:
                        edge_conf = int(edge.confidence * 100)
                        lines.append(f"      {source.name} → {edge.relation} → [{edge_conf}%]")
                if len(in_edges) > 5:
                    lines.append(f"      ... and {len(in_edges) - 5} more")
            
            shown += 1

        # Relation type summary
        if stats.relation_types:
            lines.append("\n" + "-" * 60)
            lines.append("  RELATION TYPES")
            lines.append("-" * 60)
            for rel, count in sorted(stats.relation_types.items(), key=lambda x: -x[1])[:10]:
                bar_len = min(count, 30)
                bar = "▓" * bar_len
                lines.append(f"    {rel:20s} {bar} ({count})")

        # Show orphans
        orphans = [n for n, _, _, c in nodes_with_counts if c == 0]
        if orphans:
            lines.append("\n" + "-" * 60)
            lines.append(f"  ORPHAN NODES ({len(orphans)} disconnected)")
            lines.append("-" * 60)
            for node in orphans[:15]:
                conf_pct = int(node.confidence * 100)
                type_info = node.properties.get('is_a', '')
                if isinstance(type_info, list):
                    type_info = type_info[0] if type_info else ''
                type_str = f" ({type_info})" if type_info else ""
                lines.append(f"    • {node.name}{type_str} [{conf_pct}%]")
            if len(orphans) > 15:
                lines.append(f"    ... and {len(orphans) - 15} more")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def visualize_interactive(
        self,
        output_path: str = None,
        open_browser: bool = True,
        height: str = "800px",
        width: str = "100%",
        show_edges: bool = True,
    ) -> str:
        """
        Generate an interactive HTML visualization of the knowledge graph.

        Uses pyvis to create a zoomable, draggable graph visualization.

        Args:
            output_path: Where to save the HTML file (default: {name}_graph.html)
            open_browser: Whether to open the file in the default browser
            height: Height of the visualization
            width: Width of the visualization
            show_edges: Whether to show edges (False for cluster-only view)

        Returns:
            Path to the generated HTML file.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError("pyvis required for interactive visualization. pip install pyvis")

        output_path = output_path or f"./{self.name.lower()}_graph.html"

        # Create network with physics for nice layouts
        net = Network(
            height=height,
            width=width,
            bgcolor="#1a1a2e",
            font_color="white",
            directed=True,
            select_menu=True,
            filter_menu=True,
        )

        # Disable physics - we set initial positions by cluster
        net.set_options("""
        {
            "physics": {
                "enabled": false
            },
            "nodes": {
                "font": {"size": 14},
                "scaling": {"min": 10, "max": 40},
                "chosen": {
                    "node": false
                }
            },
            "edges": {
                "color": {"inherit": true},
                "smooth": {"type": "continuous"},
                "chosen": false
            },
            "interaction": {
                "hover": true,
                "navigationButtons": true,
                "keyboard": true,
                "dragNodes": true,
                "selectConnectedEdges": false
            }
        }
        """)

        # Calculate connection counts
        conn_counts = {}
        for node in self.graph.nodes():
            out_edges = self.graph.find_edges(source_id=node.id)
            in_edges = self.graph.find_edges(target_id=node.id)
            conn_counts[node.id] = len(out_edges) + len(in_edges)

        # Check for stored embedding-based clusters first, fall back to Louvain
        node_list = list(self.graph.nodes())
        edge_list = list(self.graph.edges())

        # Try to use stored cluster_id from node properties (from /cluster command)
        node_to_cluster = {}
        has_stored_clusters = False
        for node in node_list:
            if node.properties and "cluster_id" in node.properties:
                cluster_id = node.properties["cluster_id"]
                if cluster_id >= 0:  # -1 means noise/unclustered
                    node_to_cluster[node.id] = cluster_id
                    has_stored_clusters = True

        if not has_stored_clusters:
            # Fall back to Louvain community detection
            node_to_cluster = detect_communities(node_list, edge_list)

        num_clusters = len(set(c for c in node_to_cluster.values() if c >= 0)) if node_to_cluster else 0
        cluster_colors = generate_cluster_colors(max(num_clusters, 1))

        def get_node_color(node):
            """Get node fill color based on cluster membership."""
            cluster_id = node_to_cluster.get(node.id)
            if cluster_id is not None and cluster_id < len(cluster_colors):
                return cluster_colors[cluster_id]
            # Fallback to confidence-based coloring if no cluster
            conf = node.confidence
            if conf >= 0.7:
                return "#4ecca3"  # Green
            elif conf >= 0.4:
                return "#ffd93d"  # Yellow
            else:
                return "#ff6b6b"  # Red

        def get_node_size(node, conn_count):
            """Size by connectivity (graph view) or fixed (cluster view)."""
            if show_edges:
                # Graph view: size by connectivity
                return 10 + min(conn_count * 3, 30)
            else:
                # Cluster view: fixed size for clarity
                return 20

        # Shapes without embedded labels (consistent sizing)
        cluster_shapes = [
            "dot",           # Circle
            "square",        # Square
            "diamond",       # Diamond
            "triangle",      # Triangle up
            "triangleDown",  # Triangle down
            "star",          # Star
            "hexagon",       # Hexagon
        ]

        def get_node_shape(node):
            """Get node shape based on cluster membership."""
            cluster_id = node_to_cluster.get(node.id)
            if cluster_id is not None:
                return cluster_shapes[cluster_id % len(cluster_shapes)]
            return "dot"  # Default

        def build_tooltip(node, conn_count, cluster_id):
            """Build a plain text tooltip for a node."""
            lines = []
            
            # Title with separator
            lines.append(f"━━━ {node.name} ━━━")
            lines.append("")
            
            # Stats line
            conf_pct = int(node.confidence * 100)
            stats = f"Confidence: {conf_pct}%  |  Connections: {conn_count}"
            if cluster_id is not None:
                stats += f"  |  Cluster: {cluster_id + 1}"
            lines.append(stats)
            
            # Type info
            type_info = node.properties.get("is_a")
            if type_info:
                if isinstance(type_info, list):
                    type_info = ", ".join(type_info[:3])
                lines.append(f"Type: {type_info}")
            
            # Definition - word wrap at ~50 chars
            defn = node.properties.get("definition")
            if defn:
                lines.append("")
                lines.append("Definition:")
                # Word wrap
                words = defn.split()
                current_line = "  "
                for word in words:
                    if len(current_line) + len(word) + 1 > 50:
                        lines.append(current_line)
                        current_line = "  " + word
                    else:
                        current_line += " " + word if current_line != "  " else word
                if current_line.strip():
                    lines.append(current_line)
            
            # Properties
            skip_keys = {"is_a", "definition", "is_rule", "auto_created", "from_relation"}
            other_props = [(k, v) for k, v in node.properties.items() if k not in skip_keys]
            
            if other_props:
                lines.append("")
                lines.append("Properties:")
                for k, v in other_props[:5]:
                    v_str = str(v)
                    if len(v_str) > 40:
                        v_str = v_str[:40] + "..."
                    lines.append(f"  • {k}: {v_str}")
                if len(other_props) > 5:
                    lines.append(f"  ... and {len(other_props) - 5} more")
            
            # Rule details
            if node.properties.get("is_rule"):
                lines.append("")
                lines.append("Rule:")
                condition = node.properties.get("condition", "")
                consequence = node.properties.get("consequence", "")
                lines.append(f"  IF: {condition[:60]}{'...' if len(condition) > 60 else ''}")
                lines.append(f"  THEN: {consequence[:60]}{'...' if len(consequence) > 60 else ''}")
                domain = node.properties.get("domain")
                if domain:
                    lines.append(f"  Domain: {domain}")
            
            # Metadata
            lines.append("")
            grain = node.temporal_grain.value if hasattr(node.temporal_grain, 'value') else str(node.temporal_grain)
            meta = f"Temporal: {grain}"
            if node.source:
                src_type = node.source.source_type.value if hasattr(node.source.source_type, 'value') else str(node.source.source_type)
                meta += f" | Source: {src_type}"
            lines.append(meta)
            
            return "\n".join(lines)

        # Calculate initial positions using graph layout algorithm
        import math
        import random
        random.seed(42)  # Reproducible layout

        # Different layouts for different views:
        # - Graph view: spring layout (minimizes edge crossings)
        # - Cluster view: 2D embedding projection (shows semantic proximity)
        
        use_spring_layout = show_edges
        spring_positions = {}
        embedding_positions = {}
        
        if use_spring_layout:
            # Spring layout for graph view
            try:
                import networkx as nx
                G = nx.Graph()
                for node in self.graph.nodes():
                    G.add_node(node.id)
                for edge in self.graph.edges():
                    if G.has_node(edge.source_id) and G.has_node(edge.target_id):
                        G.add_edge(edge.source_id, edge.target_id)
                spring_positions = nx.spring_layout(G, k=20.0, iterations=150, seed=42, scale=20000)
            except ImportError:
                use_spring_layout = False
        else:
            # Position clusters by semantic similarity, nodes spread within
            try:
                import numpy as np
                import json
                
                # Load embeddings from cache
                emb_cache = {}
                cache_path = os.path.join(os.path.dirname(self.save_path), "embedding_cache.json")
                if os.path.exists(cache_path):
                    with open(cache_path) as f:
                        cache_data = json.load(f)
                        emb_cache = cache_data.get("embeddings", {})
                
                # Group nodes by cluster and collect embeddings
                cluster_nodes = {}
                cluster_embeddings = {}
                for node in node_list:
                    cluster_id = node_to_cluster.get(node.id, -1)
                    if cluster_id not in cluster_nodes:
                        cluster_nodes[cluster_id] = []
                        cluster_embeddings[cluster_id] = []
                    cluster_nodes[cluster_id].append(node)
                    
                    # Get embedding for this node
                    node_type = node.properties.get("is_a", "")
                    if isinstance(node_type, list):
                        node_type = node_type[0] if node_type else ""
                    key = f"{node.name} is a {node_type}" if node_type else node.name
                    emb = emb_cache.get(key) or emb_cache.get(node.name)
                    if emb:
                        cluster_embeddings[cluster_id].append(emb)
                
                # Compute cluster centroids (average embedding)
                cluster_centroids = {}
                for cluster_id, embs in cluster_embeddings.items():
                    if cluster_id >= 0 and embs:
                        cluster_centroids[cluster_id] = np.mean(embs, axis=0)
                
                # Project cluster centroids to 2D using PCA
                cluster_ids = sorted([c for c in cluster_centroids.keys()])
                if len(cluster_ids) >= 2:
                    centroid_matrix = np.array([cluster_centroids[c] for c in cluster_ids])
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2, random_state=42)
                    cluster_2d = pca.fit_transform(centroid_matrix)
                    # Scale for good separation
                    cluster_2d = cluster_2d * 9000
                    
                    cluster_centers = {}
                    for i, cluster_id in enumerate(cluster_ids):
                        cluster_centers[cluster_id] = (float(cluster_2d[i, 0]), float(cluster_2d[i, 1]))

                    # Repulsion pass: push apart clusters that are too close
                    min_cluster_dist = 1500  # Minimum distance between cluster centers
                    for _ in range(50):  # Iterations
                        moved = False
                        for i, c1 in enumerate(cluster_ids):
                            for c2 in cluster_ids[i+1:]:
                                x1, y1 = cluster_centers[c1]
                                x2, y2 = cluster_centers[c2]
                                dx, dy = x2 - x1, y2 - y1
                                dist = math.sqrt(dx*dx + dy*dy)
                                if dist < min_cluster_dist and dist > 0:
                                    # Push apart
                                    overlap = (min_cluster_dist - dist) / 2
                                    nx, ny = dx/dist, dy/dist
                                    cluster_centers[c1] = (x1 - nx*overlap, y1 - ny*overlap)
                                    cluster_centers[c2] = (x2 + nx*overlap, y2 + ny*overlap)
                                    moved = True
                        if not moved:
                            break
                else:
                    cluster_centers = {cluster_ids[0]: (0, 0)} if cluster_ids else {}
                
                # Position nodes by distance to their cluster centroid
                random.seed(42)
                for cluster_id, nodes in cluster_nodes.items():
                    if cluster_id < 0 or cluster_id not in cluster_centers:
                        # Unclustered nodes go far out
                        for i, node in enumerate(nodes):
                            angle = (i / max(len(nodes), 1)) * 2 * math.pi
                            embedding_positions[node.id] = (
                                8000 * math.cos(angle),
                                8000 * math.sin(angle),
                            )
                    else:
                        cx, cy = cluster_centers[cluster_id]
                        centroid = cluster_centroids.get(cluster_id)
                        n_nodes = len(nodes)
                        cluster_radius = max(300, math.sqrt(n_nodes) * 150)
                        
                        # Compute distance from each node to cluster centroid
                        node_distances = []
                        for node in nodes:
                            node_type = node.properties.get("is_a", "")
                            if isinstance(node_type, list):
                                node_type = node_type[0] if node_type else ""
                            key = f"{node.name} is a {node_type}" if node_type else node.name
                            emb = emb_cache.get(key) or emb_cache.get(node.name)
                            
                            if emb is not None and centroid is not None:
                                # Cosine distance
                                emb_arr = np.array(emb)
                                dist = 1 - np.dot(emb_arr, centroid) / (np.linalg.norm(emb_arr) * np.linalg.norm(centroid) + 1e-8)
                            else:
                                dist = 0.5  # Default middle distance
                            node_distances.append((node, dist))
                        
                        # Normalize distances to 0-1 range
                        if node_distances:
                            min_dist = min(d for _, d in node_distances)
                            max_dist = max(d for _, d in node_distances)
                            dist_range = max_dist - min_dist if max_dist > min_dist else 1
                        
                        for i, (node, dist) in enumerate(node_distances):
                            # Normalize: 0 = closest to centroid, 1 = furthest
                            norm_dist = (dist - min_dist) / dist_range if dist_range > 0 else 0.5
                            # Position: closer to centroid = closer to center
                            r = cluster_radius * (0.1 + 0.9 * norm_dist)
                            # Random angle for spread
                            angle = random.random() * 2 * math.pi
                            embedding_positions[node.id] = (
                                cx + r * math.cos(angle),
                                cy + r * math.sin(angle),
                            )

                # Compaction pass: pull nodes toward cluster center, maintain minimum distance
                min_node_dist = 80  # Minimum distance between nodes
                for _ in range(15):  # Iterations (fewer = more relaxed)
                    moved = False
                    for cluster_id, nodes in cluster_nodes.items():
                        if cluster_id < 0 or cluster_id not in cluster_centers:
                            continue
                        cx, cy = cluster_centers[cluster_id]
                        node_ids = [n.id for n in nodes if n.id in embedding_positions]

                        # Pull toward center (gentle attraction)
                        for nid in node_ids:
                            x, y = embedding_positions[nid]
                            dx, dy = cx - x, cy - y
                            dist_to_center = math.sqrt(dx*dx + dy*dy)
                            if dist_to_center > 150:  # Only pull if far from center
                                pull = 0.03  # 3% toward center
                                embedding_positions[nid] = (x + dx*pull, y + dy*pull)
                                moved = True

                        # Push apart if too close (repulsion)
                        for i, n1 in enumerate(node_ids):
                            for n2 in node_ids[i+1:]:
                                x1, y1 = embedding_positions[n1]
                                x2, y2 = embedding_positions[n2]
                                dx, dy = x2 - x1, y2 - y1
                                dist = math.sqrt(dx*dx + dy*dy)
                                if dist < min_node_dist and dist > 0:
                                    overlap = (min_node_dist - dist) / 2
                                    nx, ny = dx/dist, dy/dist
                                    embedding_positions[n1] = (x1 - nx*overlap, y1 - ny*overlap)
                                    embedding_positions[n2] = (x2 + nx*overlap, y2 + ny*overlap)
                                    moved = True
                    if not moved:
                        break

            except ImportError:
                pass  # Fall back to random positioning

        

        def get_node_position(node):
            """Get initial x,y position based on view type."""
            # Graph view: spring layout
            if use_spring_layout and node.id in spring_positions:
                pos = spring_positions[node.id]
                return (float(pos[0]), float(pos[1]))
            
            # Cluster view: embedding projection
            if node.id in embedding_positions:
                return embedding_positions[node.id]
            
            # Fallback: random position
            angle = random.random() * 2 * math.pi
            radius = random.random() * 2000
            return (radius * math.cos(angle), radius * math.sin(angle))

        # Add nodes
        for node in self.graph.nodes():
            tooltip = build_tooltip(node, conn_counts[node.id], node_to_cluster.get(node.id))

            # Use cluster color for fill, confidence for border
            fill_color = get_node_color(node)
            border = get_confidence_border(node.confidence)

            x, y = get_node_position(node)
            net.add_node(
                node.id,
                label=node.name,
                title=tooltip,
                color={"background": fill_color, "border": border["color"]},
                borderWidth=border["width"],
                size=get_node_size(node, conn_counts[node.id]),
                shape=get_node_shape(node),
                x=x,
                y=y,
            )

        # Add edges (unless cluster-only view)
        if show_edges:
            relation_colors = {
                "is_a": "#74b9ff",
                "part_of": "#a29bfe",
                "has": "#fd79a8",
                "causes": "#ff7675",
                "requires": "#fdcb6e",
                "related_to": "#81ecec",
            }

            for edge in self.graph.edges():
                color = relation_colors.get(edge.relation, "#636e72")
                net.add_edge(
                    edge.source_id,
                    edge.target_id,
                    title=edge.relation,
                    label=edge.relation,
                    color=color,
                    arrows="to",
                )

        # Build full node info for detail panel
        import json as json_module
        node_details = {}
        for node in self.graph.nodes():
            details = {
                "name": node.name,
                "confidence": f"{int(node.confidence * 100)}%",
                "connections": conn_counts[node.id],
                "cluster": node_to_cluster.get(node.id),
            }
            if node.properties.get("definition"):
                details["definition"] = node.properties["definition"]
            if node.properties.get("is_a"):
                is_a = node.properties["is_a"]
                details["type"] = ", ".join(is_a) if isinstance(is_a, list) else is_a
            skip_keys = {"is_a", "definition", "is_rule", "auto_created", "from_relation", "cluster_id"}
            other_props = {k: v for k, v in node.properties.items() if k not in skip_keys}
            if other_props:
                details["properties"] = other_props
            node_details[node.id] = details

        # Generate HTML
        net.save_graph(output_path)

        # Inject detail panel and click handler
        with open(output_path, 'r') as f:
            html = f.read()

        detail_panel = """
<div id="nodeDetailPanel" style="
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    max-height: 40vh;
    overflow-y: auto;
    background: #1a1a2e;
    color: white;
    padding: 20px;
    border-top: 2px solid #4ecca3;
    font-family: monospace;
    font-size: 14px;
    display: none;
    z-index: 1000;
">
    <button onclick="document.getElementById('nodeDetailPanel').style.display='none'"
            style="float:right; background:#ff6b6b; border:none; color:white; padding:5px 10px; cursor:pointer;">
        Close
    </button>
    <div id="nodeDetailContent"></div>
</div>
<script>
var nodeDetails = """ + json_module.dumps(node_details) + """;

network.on("click", function(params) {
    if (params.nodes.length > 0) {
        var nodeId = params.nodes[0];
        var details = nodeDetails[nodeId];
        if (details) {
            var panel = document.getElementById("nodeDetailContent");
            panel.textContent = "";
            
            var h2 = document.createElement("h2");
            h2.style.marginTop = "0";
            h2.style.color = "#4ecca3";
            h2.textContent = details.name;
            panel.appendChild(h2);
            
            var p1 = document.createElement("p");
            p1.textContent = "Confidence: " + details.confidence + " | Connections: " + details.connections;
            if (details.cluster !== null && details.cluster !== undefined) {
                p1.textContent += " | Cluster: " + (details.cluster + 1);
            }
            panel.appendChild(p1);
            
            if (details.type) {
                var p2 = document.createElement("p");
                var b = document.createElement("strong");
                b.textContent = "Type: ";
                p2.appendChild(b);
                p2.appendChild(document.createTextNode(details.type));
                panel.appendChild(p2);
            }
            
            if (details.definition) {
                var p3 = document.createElement("p");
                var b2 = document.createElement("strong");
                b2.textContent = "Definition: ";
                p3.appendChild(b2);
                p3.appendChild(document.createTextNode(details.definition));
                panel.appendChild(p3);
            }
            
            if (details.properties) {
                var h3 = document.createElement("h3");
                h3.style.color = "#ffd93d";
                h3.textContent = "Properties";
                panel.appendChild(h3);
                var ul = document.createElement("ul");
                for (var key in details.properties) {
                    var li = document.createElement("li");
                    var bk = document.createElement("strong");
                    bk.textContent = key + ": ";
                    li.appendChild(bk);
                    var val = details.properties[key];
                    if (typeof val === "object") val = JSON.stringify(val);
                    li.appendChild(document.createTextNode(val));
                    ul.appendChild(li);
                }
                panel.appendChild(ul);
            }
            
            document.getElementById("nodeDetailPanel").style.display = "block";
        }
    }
});
</script>
"""
        html = html.replace("</body>", detail_panel + "</body>")

        with open(output_path, 'w') as f:
            f.write(html)

        # Open in browser if requested
        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(output_path)}")

        return output_path


def chat():
    """Interactive chat session with a tiny mind."""
    print("=" * 50)
    print("  TINY MIND - A Baby Intelligence")
    print("=" * 50)
    print()

    # Determine provider and models from environment or defaults
    provider = os.environ.get("TINYMIND_PROVIDER", "openai")
    model = os.environ.get("TINYMIND_MODEL", "gpt-4o")
    critic_model = os.environ.get("TINYMIND_CRITIC_MODEL", "gpt-4o-mini")
    use_critic = os.environ.get("TINYMIND_USE_CRITIC", "true").lower() == "true"

    # Check for API key
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set.")
        print("Set it with: export ANTHROPIC_API_KEY=your-key")
        print()
    elif provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set.")
        print("Set it with: export OPENAI_API_KEY=your-key")
        print()

    print(f"Using: {provider}/{model}")
    if use_critic:
        print(f"Critic: {provider}/{critic_model}")
    print()

    mind = TinyMind(
        name="Tiny",
        llm_provider=provider,
        model=model,
        critic_provider=provider,
        critic_model=critic_model,
        use_critic=use_critic,
    )
    print(mind.greet())
    print()

    print("Commands:")
    print("  /know [topic] - What do I know?")
    print("  /reflect     - Reflect on knowledge")
    print("  /ask         - Ask a question")
    print("  /forget X    - Forget something")
    print("  /read <path> - Read a PDF file")
    print("  /revise      - Run maintenance (dedupe, resolve contradictions)")
    print("  /audit       - Deep audit for misplaced nodes (dry run)")
    print("  /cluster     - Cluster nodes by embedding similarity")
    print("  /wonder      - What am I curious about?")
    print("  /explore     - Investigate top curiosity")
    print("  /ponder N    - N cycles of autonomous exploration")
    print("  /research X  - Research a specific topic")
    print("  /save        - Save state")
    print("  /viz         - Visualize graph (text)")
    print("  /viz-html    - Interactive graph in browser")
    print("  /quit        - Exit")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            mind.save()
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd == "/quit":
                print("Goodbye!")
                mind.save()
                break
            elif cmd == "/know":
                print(mind.know(arg))
            elif cmd == "/reflect":
                print(mind.reflect())
            elif cmd == "/ask":
                print(mind.ask(arg or ""))
            elif cmd == "/forget" and arg:
                print(mind.forget(arg))
            elif cmd == "/read" and arg:
                if arg.endswith('.pdf'):
                    try:
                        result = mind.read_pdf(arg)
                        print(f"Finished reading. Graph now has {len(list(mind.graph.nodes()))} nodes.")
                    except FileNotFoundError:
                        print(f"File not found: {arg}")
                    except Exception as e:
                        print(f"Error reading PDF: {e}")
                else:
                    print("Currently only PDF files are supported (.pdf)")
            elif cmd == "/revise":
                result = mind.revise()
                mind.save()
                print("Revised and saved!")
            elif cmd == "/audit":
                from tiny_mind.revision.reviser import Reviser
                reviser = Reviser()
                results = reviser.deep_audit(mind.graph, dry_run=True)
                if results:
                    print(f"\n(dry run - use CLI 'python -m tiny_mind audit --apply' to make changes)")
            elif cmd == "/cluster":
                from tiny_mind.revision.reviser import Reviser
                reviser = Reviser()
                node_to_cluster = reviser.cluster_by_embeddings(mind.graph)
                if node_to_cluster:
                    mind.save()
                    print("Clustered and saved! Run /viz-html to see.")
            elif cmd == "/wonder":
                goals = mind.wonder(limit=5)
                if goals:
                    print("\nWhat I'm curious about:")
                    for i, goal in enumerate(goals, 1):
                        print(f"  {i}. {goal}")
                else:
                    print("I'm not particularly curious about anything right now.")
            elif cmd == "/explore":
                result = mind.explore()
                if result.success:
                    mind.save()
            elif cmd.startswith("/ponder"):
                cycles = 3
                if arg:
                    try:
                        cycles = int(arg)
                    except ValueError:
                        pass
                mind.ponder(cycles=cycles)
                mind.save()
            elif cmd == "/research":
                if not arg:
                    print("Usage: /research <topic>")
                else:
                    result = mind.research(arg)
                    if result.success:
                        mind.save()
                        print("Saved!")
                    else:
                        print(f"Research failed: {result.errors}")
            elif cmd == "/save":
                mind.save()
                print("Saved!")
            elif cmd == "/viz":
                print(mind.visualize())
            elif cmd == "/viz-html":
                try:
                    path = mind.visualize_interactive()
                    print(f"Generated: {path}")
                except ImportError as e:
                    print(f"Error: {e}")
            else:
                print(f"Unknown command: {cmd}")
        else:
            # Regular input - learn from it
            response = mind.hear(user_input)
            print(f"\n{mind.name}: {response}\n")


if __name__ == "__main__":
    chat()
