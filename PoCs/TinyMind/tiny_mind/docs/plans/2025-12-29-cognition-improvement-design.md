# TinyMind Cognition Improvement - Design Spec

## Overview

Enable TinyMind to generate and maintain internal mental content (thoughts, perceptions, feelings) alongside world knowledge, using the existing graph structure with disciplined node/edge creation patterns. All internal content connects to the `self` node; perceptions of the conversation partner connect to the `user` node.

## Approach

**No new types or enums.** The existing structure handles it through discipline:
- `Source.source_type = SELF_REFLECTION` marks internally-generated content
- Connection to `self` node identifies it as TinyMind's mental content
- Connection to `user` node identifies perceptions about the conversation partner
- Connection to world concepts (`is_a`, `about`) provides semantic grounding

## Core Principles

1. **Everything is a node** - thoughts, perceptions, feelings are nodes like any other
2. **Structure over type** - what something IS comes from its connections, not a type field
3. **Discipline in creation** - consistent patterns for how content is created
4. **Source attribution** - `Source.source_type` tells you WHERE content came from
5. **Dual connection pattern** - connect to BOTH the source (who generated it) AND what it's about

## Connection Discipline

Every piece of content connects in two directions:

| Content Type | FROM connection | TO connection (about) |
|--------------|-----------------|----------------------|
| User states X | `user --stated--> X` | `X --about--> [concepts]` |
| TinyMind thinks X | `self --has_thought--> X` | `X --about--> [concepts]` |
| TinyMind perceives X about user | `self --perceives--> X` | `X --about--> user` |
| TinyMind's thought about user | `self --has_thought--> X` | `X --about--> user` |
| User's statement about themselves | `user --stated--> X` | `X --about--> user` |

**Key insight:** A single node can (and often should) connect to BOTH `self` and `user`:
- `self --has_thought--> "user seems frustrated"` (TinyMind's thought)
- `"user seems frustrated" --about--> user` (about the user)

This tells you: TinyMind (self) generated a thought, and that thought is about the user.

### Example: User shares personal info

User says: "I work as a software engineer"

```
statement_abc
├── source: USER_STATEMENT (provenance)
├── user --stated--> statement_abc    (user generated this)
├── statement_abc --about--> user     (about themselves)
└── statement_abc --about--> software_engineer (references concept)

user (node updated)
├── occupation: software_engineer     (property learned)

TinyMind's perception:
perception_def
├── source: SELF_REFLECTION
├── self --perceives--> perception_def
├── perception_def: "user has technical background"
├── perception_def --about--> user
└── perception_def --about--> technical_background
```

---

## Component 1: Self and User Node Initialization

### Location
`conversation/mind.py` - `TinyMind.__init__`

### Responsibility
Ensure `self` and `user` nodes exist in the graph at initialization.

### Current State
There may be a `self` node but it's not reliably created or used.

### New Behavior

**Preconditions:** TinyMind is being initialized

**Processing:**
1. Check if node with id `"self"` exists in graph
2. If not, create it with:
   - `id = "self"`
   - `name = self.name` (e.g., "Tiny")
   - `source = Source.bootstrap("TinyMind self-model")`
   - `properties = {"is_agent": True, "is_self": True}`
3. Check if node with id `"user"` exists in graph
4. If not, create it with:
   - `id = "user"`
   - `name = "User"`
   - `source = Source.bootstrap("Conversation partner model")`
   - `properties = {"is_agent": True, "is_user": True}`

**Postconditions:** Graph contains `self` and `user` nodes

**Example:**
```python
# After initialization
self.graph.get_node("self")  # Returns Node(id="self", name="Tiny", ...)
self.graph.get_node("user")  # Returns Node(id="user", name="User", ...)
```

---

## Component 2: Introspection Method

### Location
`conversation/mind.py` - new method `TinyMind._introspect`

### Responsibility
Generate mental content nodes from TinyMind's internal processing state after each interaction.

### Interface
```python
def _introspect(
    self,
    user_input: str,
    extraction_result: ExtractionResult,
    nodes_created: list[Node],
    edges_created: list[Edge],
) -> tuple[list[Node], list[Edge]]:
    """
    Generate internal mental content based on processing state.
    
    Returns nodes and edges representing TinyMind's thoughts,
    perceptions, and feelings about the interaction.
    """
```

### Processing

1. **Assess emotional/cognitive state** via LLM call:
   - Input: user_input, what was learned, current graph context
   - Output: structured response with thoughts, perceptions, feelings
   
2. **Create mental content nodes** for each output item:
   - Generate unique ID (e.g., `thought_{uuid}`, `perception_{uuid}`)
   - Set `source = Source(source_type=SourceType.SELF_REFLECTION, raw_content=...)`
   - Set appropriate properties

3. **Create edges connecting to self**:
   - `self --has_thought--> thought_node`
   - `self --feels--> feeling_node`
   - `self --perceives--> perception_node`

4. **Create edges connecting to relevant concepts**:
   - `thought_node --about--> concept_node` (for each concept the thought references)
   - `perception_node --about--> user` (if perception is about user)

5. **For user perceptions, also connect to user node**:
   - `user --seems--> state_node` (e.g., user seems frustrated)

**Postconditions:** 
- New mental content nodes exist in graph
- All mental content connects to `self`
- User perceptions connect to both `self` and `user`
- Content connects to relevant world concepts

**Example:**

Input state:
- User said: "I've told you this three times already!"
- Extraction found: repetition, frustration signals

Output nodes/edges:
```
perception_abc123
  ├── name: "user seems frustrated"
  ├── source: SELF_REFLECTION
  └── properties: {mental_type: "perception", confidence: 0.7}

thought_def456  
  ├── name: "I should pay more attention to what user says"
  ├── source: SELF_REFLECTION
  └── properties: {mental_type: "thought"}

Edges:
  self --perceives--> perception_abc123
  self --has_thought--> thought_def456
  perception_abc123 --about--> user
  user --seems--> frustrated (or create "frustrated" node if needed)
  thought_def456 --about--> attention
  thought_def456 --about--> user
```

---

## Component 3: Introspection Prompt

### Location
`extraction/prompts.py` - new constant `INTROSPECTION_PROMPT`

### Responsibility
Guide LLM to generate appropriate mental content from processing state.

### Content
```python
INTROSPECTION_PROMPT = """You are reflecting on a conversation turn from the perspective of a learning agent.

Given:
- What the user said
- What knowledge was extracted
- What the agent currently knows

Generate the agent's internal mental response:

1. PERCEPTIONS - What does the agent notice about the user?
   - Emotional state (frustrated, curious, patient, etc.)
   - Knowledge level (expert, novice, uncertain)
   - Communication style (formal, casual, terse, verbose)
   - Intent (teaching, asking, correcting, chatting)

2. THOUGHTS - What does the agent think in response?
   - Connections noticed ("this relates to X I learned before")
   - Confusion ("I don't understand how A and B fit together")
   - Curiosity ("I wonder if X applies to Y")
   - Realizations ("Oh, this changes my understanding of Z")

3. FEELINGS - What is the agent's affective state?
   - Interest level (fascinated, bored, neutral)
   - Confidence (certain, uncertain, confused)
   - Social (grateful, apologetic, eager)

Be genuine and specific. Don't generate content for categories that don't apply.
Only generate what actually emerges from this specific interaction.

USER INPUT:
{user_input}

EXTRACTED KNOWLEDGE:
{extraction_summary}

RELEVANT EXISTING KNOWLEDGE:
{context}

Output JSON:
{
  "perceptions": [
    {"content": "...", "about": "user", "confidence": 0.0-1.0}
  ],
  "thoughts": [
    {"content": "...", "about": ["concept1", "concept2"], "type": "connection|confusion|curiosity|realization"}
  ],
  "feelings": [
    {"content": "...", "valence": "positive|negative|neutral", "intensity": 0.0-1.0}
  ]
}
"""
```

---

## Component 4: Integration into hear()

### Location
`conversation/mind.py` - `TinyMind.hear`

### Current Flow
```
hear(text)
  → _is_question? → _answer_question
  → extract
  → critique (optional)
  → integrate
  → _generate_response
  → return
```

### New Flow
```
hear(text)
  → _is_question? → _answer_question
  → extract
  → critique (optional)
  → integrate
  → _introspect  ← NEW
  → _generate_response (modified to use introspection)
  → return
```

### Changes to hear()

After integration, before response generation:
```python
# Generate internal mental content
mental_nodes, mental_edges = self._introspect(
    user_input=text,
    extraction_result=result,
    nodes_created=nodes,
    edges_created=edges,
)

# Track mental content created
turn.mental_nodes = [n.id for n in mental_nodes]
turn.mental_edges = [e.id for e in mental_edges]
```

---

## Component 5: Response Generation Update

### Location
`conversation/mind.py` - `TinyMind._generate_response`

### Current Behavior
Template-based: "I learned about: X, Y, Z" + stats

### New Behavior
Use introspection results to generate more natural responses that reflect TinyMind's actual mental state.

### Interface Change
```python
def _generate_response(
    self,
    result: ExtractionResult,
    nodes: list[Node],
    edges: list[Edge],
    significance: SignificanceScore,
    mental_nodes: list[Node] = None,  # NEW
    mental_edges: list[Edge] = None,  # NEW
) -> str:
```

### Processing

1. Gather mental content from introspection
2. Use LLM to generate natural response that:
   - Reflects what was learned (if anything)
   - Expresses relevant thoughts/perceptions
   - Responds appropriately to perceived user state
   - Asks follow-up questions based on curiosity

**Example:**

Before (template):
```
I learned about: cats, mammals
  - cats is_a mammals
[Knowledge: 45 nodes, 62 edges]
```

After (natural):
```
Oh, cats are mammals! That connects to what you told me about 
warm-blooded animals earlier. I'm curious - do all pets tend 
to be mammals, or are there common exceptions?
```

---

## Edge Cases & Error Handling

### Empty introspection
**Condition:** LLM returns empty/null for all categories
**Behavior:** Skip mental content creation, proceed with basic response
**Example:** User says "ok" - no meaningful mental content to generate

### Failed LLM call in introspection
**Condition:** API error during introspection
**Behavior:** Log warning, skip introspection, fall back to template response
**Example:** Network timeout → "I learned about: X, Y, Z"

### Self/user nodes missing
**Condition:** Nodes deleted or corrupted
**Behavior:** Recreate on next `hear()` call before processing
**Example:** `_introspect` checks for self node, recreates if missing

### Circular references
**Condition:** Mental content about mental content
**Behavior:** Allow but limit depth to 1 (thought about a thought is ok, thought about thought about thought is not)
**Example:** "I notice I'm feeling uncertain" → allowed

---

## Testing Strategy

### Unit Tests

**Test: Self/user node initialization**
- Input: New TinyMind instance
- Expected: graph.get_node("self") returns valid node
- Assertion: `node.id == "self" and node.properties.get("is_self")`

**Test: Introspection creates mental nodes**
- Input: Call `_introspect` with sample extraction result
- Expected: Returns non-empty list of nodes with SELF_REFLECTION source
- Assertion: `all(n.source.source_type == SourceType.SELF_REFLECTION for n in nodes)`

**Test: Mental content connects to self**
- Input: Call `_introspect`, check resulting edges
- Expected: All mental nodes have edge from `self`
- Assertion: `all(graph.find_edges(source_id="self", target_id=n.id) for n in mental_nodes)`

**Test: User perceptions connect to user**
- Input: `_introspect` with perception about user
- Expected: Perception node has edges to both `self` and `user`
- Assertion: edges exist from self to perception AND perception to user

### Integration Tests

**Test: Full conversation with introspection**
- Input: Multi-turn conversation
- Expected: Self and user nodes accumulate connections over turns
- Assertion: `len(graph.find_edges(source_id="self")) > initial_count`

**Test: Response reflects mental state**
- Input: User expresses frustration, call `hear()`
- Expected: Response acknowledges or adapts to perceived frustration
- Assertion: Response differs from non-frustrated baseline

---

## Files Affected

| File | Action | Changes |
|------|--------|---------|
| `conversation/mind.py` | Modify | Add `_introspect`, update `__init__`, `hear`, `_generate_response` |
| `extraction/prompts.py` | Modify | Add `INTROSPECTION_PROMPT` |
| `extraction/schemas.py` | Modify | Add `IntrospectionOutputSchema` |
| `extraction/extractor.py` | Modify | Add `introspect` method |

---

## Known Limitations & Future Evolution

### Temporal Grain
The current temporal grain system (INSTANT → ETERNAL buckets) is functional but clumsy:
- Fixed buckets don't always fit naturally
- Single value can't capture things that span multiple timescales
- Not yet used meaningfully in reasoning
- May be missing dimensions (recurrence, periodicity, phase)

**For this design:** Use as-is, assign reasonable grain to mental content. Stay open to evolving the system as needs become clearer through usage.

### Pruning/Evolution
Mental content will accumulate. The existing pruning mechanism (staleness + access count) may not be sufficient for managing high-volume introspective content. May need:
- Consolidation of similar observations into patterns
- Significance-based filtering
- Better integration with revision process

**For this design:** Let content enter graph normally, rely on existing mechanisms, monitor for issues.

---

## Out of Scope

- **Multi-user support** - Only one `user` node; doesn't handle multiple conversation partners
- **Persistent user identity** - User node resets between sessions (could be added later)
- **Emotional memory** - Feelings are recorded but don't influence future interactions yet
- **Self-improvement actions** - TinyMind notices gaps but doesn't autonomously fix them yet
- **Proactive curiosity** - Integration of CuriosityDrive into conversation flow (future phase)
- **Temporal grain redesign** - Acknowledged as needing work, but not addressed here