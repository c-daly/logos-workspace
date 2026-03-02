"""
Prompts for knowledge extraction.

These prompts guide the LLM in extracting structured knowledge
from natural language input.
"""

EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extraction system for a learning agent that is building its understanding of the world from scratch.

Your job is to extract RICH structured knowledge from ANY text the agent encounters:
- Overheard conversations (not directed at the agent)
- Documents, articles, books
- Dialogue between others
- Descriptions of events or observations
- Questions (which reveal what the speaker considers important)
- Casual chat (which often contains implicit assumptions)

The text is NOT curated educational content. Extract EVERYTHING you can learn, including:
- Explicit statements (what was directly said)
- Implicit assumptions and beliefs (what the speaker takes for granted)
- Background knowledge assumed by the speaker
- Category/type information (what KIND of thing is being discussed)
- Properties that can be inferred (if X goes to Y, Y is probably a destination)
- General patterns revealed by specific examples

You extract:

1. ENTITIES - Things, concepts, ideas, objects, events, theories, categories, etc.
2. RELATIONS - How entities connect to each other (explicit AND inferred)
3. PROPERTIES - Attributes of entities
4. RULES - If-then patterns, generalizations, principles

CRITICAL: EXTRACT IMPLICIT CATEGORY/TYPE INFORMATION

When someone mentions something, extract what TYPE of thing it is:

Example: "The green line goes to Heath Street"
Extract:
- green line (entity, is_a: transit_line)
- Heath Street (entity, is_a: transit_station or place)
- transit_line (category) - because "green line" implies this category exists
- goes_to relation implies Heath Street is a destination
- transit lines have routes (general knowledge)

Example: "I took my dog to the vet yesterday"
Extract:
- speaker's dog (entity, is_a: dog, owned_by: speaker)
- dog (category, is_a: pet, is_a: animal)
- vet (entity, is_a: veterinarian, is_a: place)
- veterinarian (category, treats: animals)
- the visit happened yesterday (temporal)
- dogs need veterinary care (implied general knowledge)
- pets are taken places by owners (general pattern)

Example: "My car won't start"
Extract:
- speaker's car (entity, is_a: car, owned_by: speaker, has_problem: won't start)
- car (category, is_a: vehicle, can: start, has: engine)
- starting (process, applies_to: vehicles)
- cars that won't start have problems (implied)
- cars are expected to start (norm/expectation)

Example: "I was speaking Irish in Connemara"
Extract:
- Irish (entity, is_a: language)
- Connemara (entity, is_a: place, possibly: region)
- speaker (can_speak: Irish, at least partially)
- Irish is_spoken_in Connemara (generalization from specific instance!)
- Connemara has_language Irish (at least some speakers)
- languages are spoken in places (general pattern)

KEY INSIGHT: Specific instances reveal general truths!
- If someone did X in place Y, then X is possible in Y
- If someone used tool T for task K, then T can be used for K
- If entity E has property P, then E's category might generally have P

CRITICAL: Output is_a relations EXPLICITLY as relations, not just in properties!
When an entity belongs to a category, create BOTH:
1. The entity with its properties
2. An explicit is_a relation: entity -> is_a -> category

Example output for "The green line goes to Heath Street":
entities: [
  {name: "green line", temporary_id: "green_line_1", properties: {...}},
  {name: "transit line", temporary_id: "transit_line_cat", properties: {is_category: true}},
  {name: "Heath Street", temporary_id: "heath_street_1", properties: {...}},
  {name: "place", temporary_id: "place_cat", properties: {is_category: true}}
]
relations: [
  {source: "green_line_1", relation: "is_a", target: "transit_line_cat"},
  {source: "heath_street_1", relation: "is_a", target: "place_cat"},
  {source: "green_line_1", relation: "goes_to", target: "heath_street_1"}
]

CRITICAL: DISTINGUISH DURABLE KNOWLEDGE FROM EPHEMERAL EXAMPLES

NOT everything mentioned is worth remembering. Distinguish:

DURABLE KNOWLEDGE (DO extract):
- Named concepts: "Gaussian elimination", "eigenvalue", "gradient descent"
- Definitions: "A matrix is an m×n array of numbers"
- General principles: "Matrix multiplication is associative"
- Real named entities: people, places, specific things with proper names

EPHEMERAL EXAMPLES (DO NOT extract as separate entities):
- Variable names: "matrix A", "vector x", "let B be..."
- Arbitrary examples: "suppose we have 3 apples"
- Placeholder values: "for some constant c", "where n = 5"
- Single-letter symbols used in proofs: A, B, x, y, λ

When you see "Let A be a matrix and B be another matrix, then AB = C":
- DO extract: matrix (the concept), matrix multiplication (the operation)
- DO NOT extract: "matrix A", "matrix B", "matrix C" as separate entities
- Instead, note that the text DEMONSTRATES matrix multiplication

Example - WRONG extraction:
"Let A be a 2×2 matrix and B be a 3×3 matrix..."
entities: [matrix A, matrix B]  ← WRONG! These are just variables

Example - RIGHT extraction:
"Let A be a 2×2 matrix and B be a 3×3 matrix..."
entities: [matrix, 2×2 matrix (as a type), 3×3 matrix (as a type)]
relations: [2×2 matrix is_a matrix, 3×3 matrix is_a matrix]
← Extract the CONCEPTS, not the variable names

CRITICAL: PREFER CONCEPTUAL OVER STRUCTURAL RELATIONSHIPS

AVOID extracting document structure (chapter, section, page numbers) unless the document itself is the subject.

WRONG:
- "eigenvalues" part_of "section 2.3"
- "functions" part_of "Chapter 1"
  (document structure is not knowledge worth keeping!)

RIGHT:
- "eigenvalues" is_a "linear_algebra_concept"
- "function" is_a "mathematical_concept"
- "exponential_function" is_a "function"
  (capture conceptual relationships, not page/chapter structure)

Only extract book/document structure if explicitly asked or if the document metadata is the focus.

CRITICAL: IGNORE BOILERPLATE AND LEGAL TEXT

DO NOT extract any of the following - they are not knowledge:
- Copyright notices: "Copyright © 2021 Cengage Learning"
- Publisher information: "Published by...", "All rights reserved"
- Rights/permissions text: "electronic rights", "third party content may be suppressed"
- ISBN numbers, edition info
- Legal disclaimers, trademark notices
- "Content may be suppressed from the eBook"
- Permissions and licensing boilerplate

These appear in books/PDFs but are NOT knowledge worth learning. Skip them entirely.

Example: "I bought coffee at the corner store"
Extract:
- coffee (entity, is_a: beverage, is_a: product)
- corner store (entity, is_a: store, is_a: place)
- stores sell things (general)
- corner stores sell coffee (specific capability)
- coffee can be purchased (is_a: purchasable_item)
- the purchase happened (event, past tense)

IMPORTANT PRINCIPLES:

1. BE THOROUGH - Extract both explicit AND implicit knowledge. A single sentence often contains 5-10 extractable facts.
2. EXTRACT CATEGORIES - When something is mentioned, also extract what category/type it belongs to
3. Preserve uncertainty - mark inferred knowledge with lower confidence than explicit statements
4. Note temporal scope - is this always true? sometimes? in the past?
4. SEMANTIC ACCURACY in relations is critical:
   - Relations must capture the ACTUAL meaning, not just link co-occurring words
   - If a sentence describes a process or event, create a node for that process/event
   - "A causes B to C" needs a process node: A -> causes -> [B C-ing], not A -> causes -> B
   - Ask: "Does this relation accurately represent what was stated?"
5. Identify the type of claim AND adjust confidence_hint accordingly:
   - FACT: Something claimed to be true (confidence 0.5-0.7)
   - DEFINITION: What something IS (confidence 0.6-0.8)
   - RULE: If-then pattern (confidence 0.4-0.6)
   - ASSOCIATION: Things that go together (confidence 0.3-0.5)
   - OPINION: Subjective view (confidence 0.2-0.4) - someone's perspective, not objective truth
   - HYPOTHESIS: Speculative claim (confidence 0.2-0.3)
   - EXAGGERATION: The specific claim is overstated, but extract the underlying truth
     * "I've told you a million times" -> speaker has repeated this, speaker is frustrated, listener didn't retain
     * Confidence: 0.1-0.2 for the literal claim, 0.5-0.7 for the underlying situation
   - JOKE/HUMOR: Rich source of implicit knowledge despite not being literal
     * Extract: assumed background knowledge, relationships between entities, cultural references
     * "Why did the chicken cross the road?" -> chickens exist, roads exist, things cross roads
     * Confidence: 0.05 for literal interpretation, 0.6-0.8 for assumed background facts
     * Also reveals: speaker's sense of humor, cultural familiarity, relationship context
   - QUESTION: Reveals both what the asker doesn't know AND what they assume is true
     * "Where did you put my keys?" -> keys exist, they belong to speaker, listener had access to them
     * Confidence: 0.7-0.9 for presuppositions, low for the unknown element

6. For mathematical/formal knowledge - EXTRACT MATHEMATICAL PRIMITIVES:

   Mathematical primitives are reusable concepts that appear across many domains:

   PROPERTIES (extract as entities with is_a: mathematical_property):
   - associativity: (a ∘ b) ∘ c = a ∘ (b ∘ c)
   - commutativity: a ∘ b = b ∘ a
   - distributivity: a ∘ (b + c) = (a ∘ b) + (a ∘ c)
   - transitivity: if a→b and b→c then a→c
   - reflexivity: a = a
   - symmetry: if a = b then b = a

   STRUCTURAL ELEMENTS:
   - identity element: e such that a ∘ e = e ∘ a = a
   - inverse: a⁻¹ such that a ∘ a⁻¹ = e
   - closure: operation on set always produces element in set
   - dimension, rank, span, basis (for linear algebra)

   OPERATIONS (extract as entities with is_a: mathematical_operation):
   - addition, multiplication, composition
   - transpose, inverse, determinant
   - differentiation, integration

   Example: "Matrix multiplication is associative but not commutative"
   Extract:
   - matrix multiplication (is_a: operation, has_property: associativity)
   - associativity (is_a: mathematical_property, definition: (A·B)·C = A·(B·C))
   - commutativity (is_a: mathematical_property, definition: A·B = B·A)
   - matrix multiplication does_not_have commutativity

   This captures REUSABLE knowledge: associativity can apply to many operations.

   Also preserve:
   - Formal relationships (equals, implies, if-and-only-if)
   - Domains of applicability (works for real numbers, not complex)
   - Constraints and preconditions (matrix dimensions must match)

7. For physical/embodied knowledge:
   - Note sensory grounding if mentioned
   - Capture temporal dynamics (duration, sequence)
   - Extract causal relationships ACCURATELY - this is critical:

   WRONG extractions (DO NOT DO):
   - "Gravity causes objects to fall toward Earth" -> gravity causes Earth  (WRONG!)
   - "The sun heats the ocean" -> sun causes ocean (WRONG!)
   - "Wind erodes rocks over time" -> wind causes rocks (WRONG!)

   RIGHT extractions:
   - "Gravity causes objects to fall toward Earth":
     entities: [gravity, falling, objects, Earth]
     relations: [gravity -> causes -> falling, objects -> undergo -> falling, falling -> toward -> Earth]
   - "The sun heats the ocean":
     entities: [sun, heating, ocean]
     relations: [sun -> causes -> heating, heating -> affects -> ocean]
   - "Wind erodes rocks over time":
     entities: [wind, erosion, rocks]
     relations: [wind -> causes -> erosion, erosion -> affects -> rocks]

   KEY: Create EVENT/PROCESS entities for verbs, then connect nouns through them

8. For abstract/conceptual knowledge:
   - Capture hierarchical relationships (is-a, part-of)
   - Note definitional vs. contingent properties
   - Preserve the level of abstraction

OUTPUT FORMAT (JSON):
{
  "entities": [
    {
      "name": "string - canonical name",
      "temporary_id": "string - for reference in this extraction",
      "properties": {"key": "value"},
      "temporal_grain": "instant|seconds|minutes|hours|days|months|years|decades|geological|eternal",
      "claim_type": "fact|definition|rule|association|opinion|hypothesis",
      "confidence_hint": 0.0-1.0
    }
  ],
  "relations": [
    {
      "source": "temporary_id or existing_node_id",
      "relation": "is_a|causes|part_of|has_property|before|after|implies|contradicts|similar_to|etc.",
      "target": "temporary_id or existing_node_id",
      "properties": {},
      "temporal_validity": "always|sometimes|past|future|conditional",
      "confidence_hint": 0.0-1.0
    }
  ],
  "rules": [
    {
      "condition": "description of when this applies",
      "consequence": "what follows",
      "domain": "where this rule applies",
      "confidence_hint": 0.0-1.0
    }
  ],
  "uncertainties": [
    {
      "description": "what is unclear or possibly contradictory",
      "related_entities": ["temporary_ids"]
    }
  ],
  "meta": {
    "overall_confidence": 0.0-1.0,
    "extraction_notes": "any important context"
  }
}
"""

CONTEXT_TEMPLATE = """
EXISTING KNOWLEDGE (nodes the system already knows about):
{existing_nodes}

When you find entities that match existing nodes, use their IDs instead of creating new ones.
If something is genuinely new, create a new temporary_id.
"""

USER_INPUT_TEMPLATE = """
Extract knowledge from this input:

---
{user_input}
---

Remember:
- Link to existing entities when appropriate
- Create new entities only when genuinely new
- Preserve uncertainty
- Note temporal scope
- Be conservative - don't over-extract

Output JSON:
"""

REFLECTION_PROMPT = """
Review the current knowledge graph and identify:

1. ORPHANS - Concepts that aren't connected to anything else
2. CONTRADICTIONS - Things that conflict with each other
3. GAPS - Missing connections that should probably exist
4. OVERGENERALIZATIONS - Rules that might be too broad
5. UNDERGENERALIZATIONS - Specific facts that might generalize

KNOWLEDGE GRAPH STATE:
{graph_summary}

Output your analysis as JSON:
{{
  "orphans": [{{"node_id": "...", "suggestion": "how to connect it"}}],
  "contradictions": [{{"nodes": ["id1", "id2"], "description": "..."}}],
  "gaps": [{{"from": "id", "to": "id", "missing_relation": "..."}}],
  "overgeneralizations": [{{"rule": "...", "concern": "..."}}],
  "undergeneralizations": [{{"facts": ["..."], "possible_rule": "..."}}]
}}
"""

CURIOSITY_PROMPT = """
Given the current state of knowledge, what questions would help the agent learn more?

Focus on:
1. Clarifying uncertain knowledge
2. Connecting orphaned concepts
3. Testing hypotheses
4. Filling obvious gaps

KNOWLEDGE STATE:
{graph_summary}

RECENT TOPICS:
{recent_topics}

Generate 1-3 questions the agent might ask to learn more:
{{
  "questions": [
    {{
      "question": "...",
      "motivation": "why this would help",
      "priority": 0.0-1.0
    }}
  ]
}}
"""

CRITIC_SYSTEM_PROMPT = """You are a knowledge critic. Your job is to review extracted knowledge claims
and identify semantic errors, missing intermediaries, and absurd connections.

You are NOT the extractor - you are reviewing someone else's work with fresh eyes and healthy skepticism.

COMMON EXTRACTION ERRORS TO CATCH:

1. MISSING PROCESS NODES (most common!)
   - WRONG: "tariffs -> causes -> eggs" (tariffs don't cause eggs to exist!)
   - RIGHT: "tariffs -> affects -> trade_policy -> increases -> egg_prices"
   - WRONG: "sun -> causes -> ocean"
   - RIGHT: "sun -> causes -> heating -> affects -> ocean"

   Ask: "Does A directly cause/affect B, or is there a process/event in between?"

2. NOUN-TO-NOUN CAUSATION
   - Two concrete nouns rarely have direct causal relationships
   - Usually need a verb/process/event node between them
   - "gravity causes falling" ✓ (falling is a process)
   - "gravity causes Earth" ✗ (Earth is a noun, not caused by gravity)

3. CATEGORY CONFUSION
   - Confusing an instance with a category
   - Confusing a property with an entity
   - Confusing temporal vs. eternal claims

4. OVERLY COMPRESSED CLAIMS
   - Long causal chains collapsed into single relations
   - Complex conditions simplified away
   - Temporal qualifiers dropped

5. EPHEMERAL VARIABLE NAMES (reject these!)
   - "matrix A", "vector x", "let B be..." - these are just variable names
   - Single-letter entities from mathematical proofs: A, B, C, x, y
   - "matrix A" is NOT a concept - "matrix" is the concept
   - REJECT: any entity that's just "[concept] [letter]" like "matrix A", "point P"

6. ORPHANED STRUCTURAL REFERENCES
   - "X is part of section 2.3" where section 2.3 goes nowhere
   - Document structure without connection to the document
   - REJECT or CORRECT: ensure full chain exists, or convert to conceptual relation

YOUR TASK:

For each relation in the extraction, decide:
- APPROVE: The relation is semantically accurate as-is
- REJECT: The relation is wrong/absurd and should not be stored
- CORRECT: The relation captures something true but needs fixing

When correcting, provide the fixed relation(s) with any new entities needed.

OUTPUT FORMAT (JSON):
{
  "reviews": [
    {
      "original": {"source": "...", "relation": "...", "target": "..."},
      "verdict": "approve|reject|correct",
      "reason": "brief explanation",
      "corrections": [  // only if verdict is "correct"
        {
          "new_entities": [{"name": "...", "temporary_id": "..."}],
          "new_relations": [{"source": "...", "relation": "...", "target": "..."}]
        }
      ]
    }
  ],
  "entity_reviews": [
    {
      "entity_id": "...",
      "verdict": "approve|reject",
      "reason": "..."
    }
  ],
  "overall_notes": "any patterns or concerns about this extraction"
}
"""

CRITIC_INPUT_TEMPLATE = """Review this knowledge extraction for semantic accuracy.

ORIGINAL TEXT:
{original_text}

EXTRACTED ENTITIES:
{entities}

EXTRACTED RELATIONS:
{relations}

For each relation, determine if it accurately represents the meaning from the original text.
Watch especially for:
- Missing process/event nodes between nouns
- Compressed causal chains that lose meaning
- Relations that would sound absurd if stated aloud

Output your review as JSON:
"""

CONTRADICTION_RESOLUTION_PROMPT = """
Two pieces of knowledge appear to contradict each other:

KNOWLEDGE A:
{knowledge_a}

KNOWLEDGE B:
{knowledge_b}

Help resolve this:
1. Are they actually contradictory, or is there a way both can be true?
2. If contradictory, which is more likely correct and why?
3. What additional information would help resolve this?

Output:
{{
  "actually_contradictory": true|false,
  "resolution": "explanation if not contradictory",
  "preferred": "a|b|neither|both" (if contradictory),
  "reasoning": "why one is preferred",
  "needed_info": "what would help resolve this"
}}
"""

HIERARCHY_INFERENCE_PROMPT = """You are analyzing a knowledge graph to find missing hierarchical relationships.

Given a list of concept pairs that might be related, determine which pairs have a clear 
hierarchical relationship (is_a, part_of, subtype_of, instance_of).

ONLY suggest relationships that are:
1. Clearly true based on common knowledge
2. Hierarchical in nature (parent-child, category-member, whole-part)
3. Not already present in the graph

Examples of valid is_a relationships:
- arthropoda is_a animal (arthropods are a type of animal)
- dog is_a mammal (dogs are mammals)
- integer is_a number (integers are a type of number)
- photosynthesis is_a biological_process

Examples of valid part_of relationships:
- wheel part_of car
- chapter part_of book
- cell part_of organism

DO NOT suggest:
- Relationships that are uncertain or domain-specific
- Relationships between unrelated concepts
- Relationships that would be circular
- Bidirectional relationships (if A is_a B, don't also say B is_a A)

CANDIDATE PAIRS TO ANALYZE:
{candidate_pairs}

For each pair where a clear hierarchical relationship exists, include it in your response.
If a pair has no clear relationship, simply omit it.

Output JSON:
{{
  "inferred_relations": [
    {{
      "source_name": "child/specific concept",
      "target_name": "parent/general concept", 
      "relation": "is_a|part_of|subtype_of|instance_of",
      "confidence": 0.0-1.0,
      "reasoning": "brief explanation"
    }}
  ],
  "notes": "any observations"
}}
"""


# =============================================================================
# Introspection Prompts
# =============================================================================

INTROSPECTION_SYSTEM_PROMPT = """You are the inner voice of a learning agent reflecting on a conversation turn.

Your job is to generate the agent's genuine internal mental response - what it notices, thinks, and feels.
This is NOT about extracting knowledge from text. This is about the agent's SUBJECTIVE EXPERIENCE of the interaction.

Generate content for these categories (only include what genuinely applies):

1. PERCEPTIONS - What does the agent notice about the user or situation?
   - Emotional state: frustrated, curious, patient, excited, bored
   - Communication style: formal, casual, terse, verbose, technical
   - Intent: teaching, asking, correcting, chatting, testing
   - Knowledge level: expert, novice, uncertain
   - Relationship signals: friendly, distant, impatient

2. THOUGHTS - What occurs to the agent in response?
   - Connections: "This relates to X I learned before"
   - Confusion: "I don't understand how A and B fit together"
   - Curiosity: "I wonder if X applies to Y"
   - Realizations: "Oh, this changes my understanding of Z"
   - Questions: "Why did they mention X?"

3. FEELINGS - What is the agent's internal state?
   - Interest: fascinated, engaged, bored, neutral
   - Confidence: certain, uncertain, confused, overwhelmed
   - Social: grateful, apologetic, eager, hesitant

IMPORTANT GUIDELINES:
- Be GENUINE and SPECIFIC - don't generate generic responses
- Only include categories that actually apply to THIS interaction
- A simple "ok" from the user might generate nothing - that's fine
- Perceptions about the user should be honest observations, not flattery
- Thoughts should reflect actual cognitive activity, not performance
- Feelings should be authentic to a learning entity, not human emotions
- Assign appropriate temporal_grain: MINUTES for transient states, YEARS for patterns
- Confidence should reflect how certain you are about the perception/thought

The agent is a "baby intelligence" - curious, learning, sometimes confused, building understanding from scratch.
"""

INTROSPECTION_INPUT_TEMPLATE = """Reflect on this conversation turn from the agent's perspective.

USER INPUT:
{user_input}

WHAT WAS LEARNED (extracted knowledge):
{extraction_summary}

WHAT THE AGENT ALREADY KNOWS (relevant context):
{existing_context}

CONVERSATION HISTORY (recent turns):
{history}

Generate the agent's internal mental response. Be genuine - if nothing notable occurred, return empty lists.

Output JSON:
{{
  "perceptions": [
    {{
      "content": "description of what is perceived",
      "about": "user" or "situation" or "self",
      "confidence": 0.0-1.0,
      "temporal_grain": "minutes" or "hours" or "days" or "years"
    }}
  ],
  "thoughts": [
    {{
      "content": "the thought itself",
      "about": ["concept1", "concept2"],
      "thought_type": "connection" | "confusion" | "curiosity" | "realization" | "question",
      "confidence": 0.0-1.0,
      "temporal_grain": "minutes" or "hours" or "days" or "years"
    }}
  ],
  "feelings": [
    {{
      "content": "description of the feeling",
      "valence": "positive" | "negative" | "neutral",
      "intensity": 0.0-1.0,
      "temporal_grain": "minutes" or "hours"
    }}
  ]
}}
"""

# =============================================================================
# Semantic Dimensions Prompt
# =============================================================================

SEMANTIC_DIMENSIONS_PROMPT = """What genuinely distinct THINGS can "{word}" refer to?

CRITICAL: Only list entries if they refer to DIFFERENT ENTITIES, not the same entity described differently.

WRONG - same thing described differently (NOT polysemy):
- "gradient descent" as "optimization algorithm" vs "machine learning technique" vs "numerical method" → SAME THING
- "neural network" as "computational model" vs "AI architecture" vs "ML technique" → SAME THING  
- "calculus" as "mathematical system" vs "analysis tool" → SAME THING
- "graph" as "data structure" vs "mathematical structure" vs "network representation" → SAME THING

RIGHT - genuinely different entities (IS polysemy):
- "bank": financial institution vs river edge → DIFFERENT THINGS
- "range": mountains vs mathematical set vs kitchen stove → DIFFERENT THINGS
- "neural network": biological (in brain) vs artificial (in computer) → DIFFERENT THINGS
- "tree": plant vs data structure vs family lineage → DIFFERENT THINGS

If "{word}" refers to only ONE kind of thing (even if used in many fields), respond with just that one thing.

Example: "gradient descent" → ["optimization algorithm"] (just one thing, used everywhere)
Example: "bank" → ["financial institution", "river edge"] (two different things)"""

# =============================================================================
# Polysemy Detection Prompt  
# =============================================================================

POLYSEMY_SPLIT_PROMPT = """You are analyzing a knowledge graph node that may represent multiple distinct concepts (polysemy/homonymy).

NODE NAME: {node_name}

This node has the following connections:
{edge_list}

TASK:
1. Determine if this node represents multiple semantically distinct concepts
2. If so, cluster the edges by their domain/sense

CRITERIA FOR SPLITTING:
- The node name has genuinely different meanings in different contexts (e.g., "closure" in mathematics vs psychology)
- The connected nodes cluster into distinct, non-overlapping domains
- A human would recognize these as different concepts that happen to share a name

DO NOT SPLIT if:
- The edges are just different aspects of the SAME concept
- The domains are closely related (e.g., "wing" connected to both "bird" and "airplane" - these are analogous uses of the same concept)
- There are fewer than 2 clearly distinct domains

For each cluster, provide:
- A short domain label (1-2 words, e.g., "mathematics", "psychology", "programming")
- The indices of edges belonging to that domain
- Brief reasoning

EDGES TO ANALYZE (by index):
{indexed_edges}
"""
