# Intake Summary: TinyMind Cognition Improvement

## Task
Improve TinyMind's cognition to be more accurate, subtle, and autonomous.

## Success Criteria

### Accuracy (All Areas)
- [ ] Better extraction quality - fewer errors in what gets extracted
- [ ] Better reasoning - draw correct inferences, avoid false conclusions  
- [ ] Better answers - synthesize knowledge to answer questions correctly

### Subtlety (All Areas)
- [ ] Nuanced responses - acknowledge uncertainty, partial knowledge, complexity
- [ ] Conversational tone - less robotic, more natural dialogue
- [ ] Implicit understanding - read between the lines, understand context
- [ ] Emotional awareness - sense mood, adapt communication style

### Autonomy (All Areas)
- [ ] Proactive curiosity - ask follow-up questions, seek clarification
- [ ] Self-directed learning - research topics independently
- [ ] Initiative in conversation - share insights, make connections
- [ ] Self-improvement - identify and fix own knowledge gaps

## Current Architecture

```
conversation/mind.py     - TinyMind class (main interface)
extraction/extractor.py  - LLM-based knowledge extraction
extraction/prompts.py    - Extraction/critic prompts
curiosity/drive.py       - Goal generation
revision/reviser.py      - Knowledge maintenance
substrate/              - Graph storage (node, edge, graph)
```

## Key Files to Modify

| File | Current Issue | Improvement Area |
|------|---------------|------------------|
| `conversation/mind.py` | `_generate_response()` is template-based | Subtlety, Autonomy |
| `conversation/mind.py` | `_assess_significance()` uses simple heuristics | Accuracy, Subtlety |
| `conversation/mind.py` | `_answer_question()` is keyword matching | Accuracy |
| `conversation/mind.py` | `hear()` has no active reasoning step | Accuracy, Autonomy |
| `extraction/prompts.py` | Prompts don't emphasize nuance/context | Subtlety |
| `curiosity/drive.py` | Not integrated into conversation flow | Autonomy |

## Constraints
- Maintain backward compatibility with existing graph format
- Keep LLM calls efficient (avoid unnecessary API calls)
- Preserve the "learning baby" character of TinyMind
- Should work with both OpenAI and Anthropic providers

## Workflow Classification
**COMPLEX** - Multiple files, architectural decisions, comprehensive scope

## Relevant Capabilities
- Serena symbolic editing tools
- LLM for response generation improvements
- Existing curiosity/drive system to integrate

## Design Philosophy
**"Baby intelligence" = minimal structure enabling emergent growth**

NOT about maintaining naive character. The goal is:
- Define as little as possible upfront
- Create structure that allows capabilities to emerge
- Let sophistication grow from the system, not be programmed in

Key constraint: **Improvements should remove barriers to growth and connect existing systems, not add scripted behaviors.**

## Risks
- Scope creep - touches many areas
- Over-engineering - adding complexity instead of enabling emergence
- Performance - more LLM calls = more latency/cost
- Scripted sophistication - faking intelligence instead of enabling it

## Recommended Approach
Phase the work:
1. **Phase A**: Response quality (subtlety, conversational tone)
2. **Phase B**: Reasoning accuracy (inference, synthesis)
3. **Phase C**: Autonomy (curiosity integration, initiative)

This allows incremental testing and avoids big-bang risk.
