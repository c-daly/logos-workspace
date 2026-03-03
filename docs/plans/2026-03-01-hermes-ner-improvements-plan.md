# Hermes NER Quality Improvements - Implementation Plan

**Story:** [logos #502](https://github.com/c-daly/logos/issues/502) — Hermes NER Quality Improvements
**Design Reference:** [`docs/experiments/2026-03-01-rottie-clustering-handoff.md`](../experiments/2026-03-01-rottie-clustering-handoff.md)

## Context

The rottie/rottweiler experiment revealed three pre-ingestion quality problems in Hermes' NER pipeline:

1. **No name normalization** — `Rotties` and `rottie` created as separate entities. No lowercasing, no singularization, no dedup within a single proposal.
2. **No type awareness** — `ONTOLOGY_TYPES` is hardcoded in `hermes/src/hermes/ner_provider.py`. Sophia's actual type list (which evolves) is never consulted. Everything that isn't obviously a location ends up as `object`.
3. **NER on prompt only** — The `/llm` endpoint extracts entities from the user's text only. The LLM reply restates things more formally, uses canonical names, and makes implicit connections explicit — but none of that gets extracted.

Note: Sophia currently ignores the `type` field in proposals (classifies via centroid proximity). Type-aware NER still has value: it reduces noise in the `object` centroid, improves relationship extraction quality, and makes proposals more debuggable.

## Tasks

### Task 1: Entity Name Normalization

Add a post-extraction normalization step that cleans entity names before they leave Hermes. No extra LLM call — this is string processing.

**Do:**
- Create `hermes/src/hermes/name_normalizer.py` with a `normalize_entities(entities: list[dict], text: str) -> list[dict]` function that:
  - Lowercases all entity names (preserving original offsets)
  - Singularizes simple English plurals (`Rotties` → `rottie`, `checkups` → `checkup`). Use a lightweight approach (suffix rules), not a full NLP library.
  - Deduplicates entities within a single proposal — if two entities normalize to the same name, keep the one with the longer span (more context) and merge their offsets
  - Preserves the `type`, `start`, `end` fields (adjusting as needed for merges)
- Call `normalize_entities()` from `hermes/src/hermes/proposal_builder.py` in the `build()` method, after NER extraction returns but before embeddings are computed (so embeddings use normalized names)
- Create `hermes/tests/unit/test_name_normalizer.py` with tests for:
  - Lowercasing (`Rottweiler` → `rottweiler`)
  - Singularization (`Rotties` → `rottie`, `checkups` → `checkup`)
  - Deduplication within a proposal (two entities normalizing to same name)
  - Preservation of offsets and types
  - Edge cases: single-character names, already-lowercase, names that shouldn't be singularized (`Paris`, `diabetes`)

**Verify:**
- `cd hermes && poetry run pytest tests/unit/test_name_normalizer.py -v` passes
- `cd hermes && poetry run ruff check src/hermes/name_normalizer.py` clean
- `cd hermes && poetry run pytest tests/unit/test_proposal_builder.py -v` still passes (existing tests unbroken)

---

### Task 2: Type-Aware NER Prompt

Fetch Sophia's current type list and inject it into the NER prompt, replacing the hardcoded `ONTOLOGY_TYPES` dict. Falls back to the hardcoded list if Sophia is unreachable.

**Do:**
- Create `hermes/src/hermes/ontology_client.py` with:
  - `async def fetch_type_list(sophia_url: str) -> list[dict]` — calls Sophia's API to get current node types. Parse the response into `[{"name": "location", "description": "..."}, ...]`
  - `async def fetch_edge_type_list(sophia_url: str) -> list[dict]` — calls Sophia's API to get current edge types
  - Cache the results with a configurable TTL (default 5 minutes). Use a simple in-memory cache with timestamp — no Redis dependency yet.
  - On fetch failure, log a warning and return `None` (caller falls back to hardcoded types)
- Modify `hermes/src/hermes/combined_extractor.py`:
  - Change `_SYSTEM_PROMPT` from a module-level constant to a method `_build_system_prompt(self, type_list: list[dict] | None = None, edge_type_list: list[dict] | None = None) -> str`
  - If `type_list` is provided, use it instead of `ONTOLOGY_TYPES` in the prompt. Format: `"- {name}: {description}"` per type. Append: `"If none of these types fit, use 'object'."`
  - If `edge_type_list` is provided, add a section: `"## Known Relation Types\nUse these relation labels where appropriate:\n"` followed by the list. Append: `"You may also use other UPPER_SNAKE_CASE labels if none of these fit."`
  - In `extract_entities_and_relations()`, call the ontology client to get current types before building the prompt. Use Sophia URL from config (`SOPHIA_HOST`/`SOPHIA_PORT` env vars, same as existing `_get_sophia_context` uses).
- Modify `hermes/src/hermes/ner_provider.py`:
  - Apply the same dynamic prompt pattern to `OpenAINERProvider` (the fallback NER-only provider)
  - Keep `ONTOLOGY_TYPES` as the fallback default
- Create `hermes/tests/unit/test_ontology_client.py` with tests for:
  - Successful fetch returns parsed type list
  - Fetch failure returns `None` (no exception raised)
  - Cache returns stale value within TTL
  - Cache expires after TTL
- Update `hermes/tests/unit/test_combined_extractor.py` with tests for:
  - Dynamic prompt includes fetched types when available
  - Falls back to `ONTOLOGY_TYPES` when fetch returns `None`
  - Edge type list appears in prompt when available

**Verify:**
- `cd hermes && poetry run pytest tests/unit/test_ontology_client.py -v` passes
- `cd hermes && poetry run pytest tests/unit/test_combined_extractor.py -v` passes
- `cd hermes && poetry run pytest tests/unit/test_ner_provider.py -v` passes
- `cd hermes && poetry run ruff check src/hermes/ontology_client.py src/hermes/combined_extractor.py src/hermes/ner_provider.py` clean

---

### Task 3: NER on Prompt and Reply

After generating the LLM response, build a second proposal from the combined prompt + reply text and send it to Sophia. The existing pre-generation proposal (used for context retrieval) stays unchanged.

**Do:**
- Modify `hermes/src/hermes/main.py` in the `/llm` endpoint (the `llm_generate` function around line 769):
  - After `result = await generate_llm_response(...)` returns, extract the reply text from `result["choices"][0]["message"]["content"]`
  - Concatenate: `combined_text = f"{user_text}\n\n{reply_text}"`
  - Build a second proposal: `await ProposalBuilder().build(combined_text, metadata={..., "extraction_source": "prompt_and_reply"})`
  - Send this proposal to Sophia via the same mechanism as `_get_sophia_context` but fire-and-forget (don't await context back, don't block the response). Use the existing Redis enqueue path if available, or a background `asyncio.create_task` wrapping the synchronous Sophia call.
  - Add a metadata field `"extraction_source": "prompt_and_reply"` to distinguish from the pre-generation proposal which should get `"extraction_source": "user_prompt"`
  - Tag the pre-generation proposal's metadata with `"extraction_source": "user_prompt"` for clarity
- Create `hermes/tests/unit/test_llm_endpoint_dual_proposal.py` with tests for:
  - When `/llm` is called, two proposals are built (one pre-generation, one post-generation)
  - The post-generation proposal contains both user text and reply text
  - The post-generation proposal has `extraction_source: "prompt_and_reply"` in metadata
  - The response is not delayed by the second proposal (fire-and-forget)
  - If the second proposal fails, the endpoint still returns successfully

**Verify:**
- `cd hermes && poetry run pytest tests/unit/test_llm_endpoint_dual_proposal.py -v` passes
- `cd hermes && poetry run pytest tests/unit/test_main.py -v` still passes (existing endpoint tests unbroken)
- `cd hermes && poetry run ruff check src/hermes/main.py` clean
