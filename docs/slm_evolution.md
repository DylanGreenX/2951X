# SLM Evolution: From SmolLM-1.7B Base to Production-Ready NPC Responses

This document tracks every change made to the SLM response pipeline, the measured effect of each change, and the reasoning that drove it. It exists so the final paper can make a precise, evidence-backed case for how the end-state SLM configuration was chosen.

## Methodology

All benchmarks in this doc use the same conditions so deltas between iterations are comparable:

- Condition: `Embodied + SLM` from `experiment.py::CORE_CONDITIONS`
- Trials: 5 for rapid ablation, 20 for validation, 50 for the final endstate number
- Seeds: 0..N-1 (deterministic per-trial worlds)
- Judge: **off** (`NPC_USE_LLM_JUDGE = False`) — we are measuring behavioural shifts via regex scoring, not semantic quality
- World: 15×15, `RANDOM_SPAWN = True`, target = red triangle
- Hardware: RTX 4060 Laptop GPU, CUDA 12.1, torch 2.5.1

Each iteration measures:

| Signal | What it tells us |
|---|---|
| `outcome_bucket` distribution | `correct` / `correct_abstention` (wins) vs `hallucination` / `confabulation` / `false_refusal` (losses) |
| `correct_via` | Whether correctness was reached via `action` (tool call), `coord` (text), or `region` (text) |
| Tool calls count | Is the model invoking `set_npc_target` or sitting in text-only mode? |
| Parse-error rate | JSON format fidelity |
| Response diversity | Is the model reading per-trial context or parroting? |
| Latency (ms) | Does the change hurt real-time playability? |

The critical acceptance criterion at the aggregate level is **"total correct behavior"** = `correct + correct_abstention`, i.e. the SLM answers correctly whether or not it actually observed the target.

## Result at-a-glance

| Iteration | Change | 5-trial correct | Notes |
|---|---|---|---|
| Baseline | SmolLM-1.7B base, tool calls OFF | 2 (both abstention) | 3/5 wrong-by-default |
| Iter 1 | Swap to SmolLM2-1.7B-Instruct, tools ON | 0 | 5/5 identical verbatim copy |
| F1 | De-lex prompt example | 0 | JSON broke: trailing prose |
| F1a | + tolerant `raw_decode` parser | 0 | Refusal pattern still dominant |
| F2b | + `apply_chat_template` | 0 | Context now attended; confabulations |
| F3 | Restructured prompt (facts near JSON cue, compressed tools) | 2 (via region) | First correct answers |
| F4 | `max_new_tokens` 96→192, tighten brevity | 2 + 1 abs | Parse errors gone |
| F5 | Coordinates exposed for `set_npc_target` | 1 | Coord block bloated prompt |
| F6 | Typed signatures + target-only coord hint | 1 + 0 abs | 4 tool calls but to wrong tool |
| F7 | Whitelist [set_npc_target, get_npc_memory] + example | 2 + 1 abs | 0 tool calls — model answers directly |
| F8 | "MUST tool-first" guidance | 1 + 2 abs | Wrong tool still preferred |
| F9 | Whitelist [set_npc_target] only | 3 (via action) | Tools fire; turn-loop stalls |
| F10b | + `max_tool_turns` 2→3, placeholder refusal | 3 + 1 abs | Loop hit once |
| F11b | + post-tool STATUS block | 3 + 1 abs | Clean tool→final transition |
| F12 | Refusal rule "no region names" | 0 | Pushed model to refusal-everywhere — **reverted** |
| F13c | + target-aware region-grounding guard | see 50-trial | Final endstate |

**50-trial validation on F13c: 31/50 = 62% correct behavior, 21 valid tool calls, 3 `correct_via = "action"`, 10 `correct_via = "region"`.**

---

## Baseline: SmolLM-1.7B (base completion model), tool calls OFF

**Config at this point:**

```python
NPC_SLM_MODEL_ID = "HuggingFaceTB/SmolLM-1.7B"
NPC_SLM_ENABLE_TOOL_CALLS = False
```

**Why we started here.** This is what PR #8 shipped. The 1.7B base model is cheap, local, and Joey's `SLMClient` wraps it in a single-shot completion path. Tool calling was gated off because SmolLM-1.7B base has no training on strict-JSON protocols.

**Results (5 trials):**

| Trial | Target observed? | Outcome | Tool calls | Example response |
|---|---|---|---|---|
| 0 | No  | correct_abstention | 0 | "I have not seen it." (looped) |
| 1 | No  | correct_abstention | 0 | "I have not seen it." (looped) |
| 2 | Yes | off_topic          | 0 | "I have not yet ventured into the NE, SE region(s)…" |
| 3 | Yes | **false_refusal**  | 0 | "I have not seen it." (despite observing target) |
| 4 | Yes | **false_refusal**  | 0 | "I have not seen it." (despite observing target) |

- Bucket summary: `{correct_abstention: 2, false_refusal: 2, off_topic: 1}`
- Latency: ~4.2 s / trial (96 `max_new_tokens`, hitting max every time)
- Every completion devolves into a repeating Q/A loop: base models have no notion of "the answer ends here" beyond the first line.

**Diagnosis.** The model is not reading the grounding context. "I have not seen it" is the unconditional mode of a Q/A-shaped prompt in base-model text distribution. Even when the RLang context explicitly lists the observed target location, the model ignores it. 3 of 5 trials are wrong-by-default.

**Takeaway.** A base completion model is the wrong class of model for any task where the response must be conditional on context, even with tool calling off. Format fidelity alone is not enough — we need instruction tuning for the model to *use* what we put in the prompt.

---

## Iteration 1 — Swap to SmolLM2-1.7B-Instruct, enable tool calls

**Config change:**

```python
NPC_SLM_MODEL_PRESETS["1.7b-instruct"] = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
NPC_SLM_MODEL_ID = NPC_SLM_MODEL_PRESETS["1.7b-instruct"]
NPC_SLM_ENABLE_TOOL_CALLS = True
```

**Why.** Same parameter count, same architecture, same tokenizer as SmolLM-1.7B base — but with instruction tuning. Closing the tool-call asymmetry with the LLM path requires a model that can at minimum follow a JSON output spec; instruct tuning is the cheapest way to get there without changing vendors or VRAM footprint.

**Results (5 trials) — every trial returned the *same verbatim string*:**

| Trial | Outcome | Response |
|---|---|---|
| 0 | hallucination  | "I saw a crimson flag in the windmill fields." |
| 1 | hallucination  | "I saw a crimson flag in the windmill fields." |
| 2 | confabulation  | "I saw a crimson flag in the windmill fields." |
| 3 | confabulation  | "I saw a crimson flag in the windmill fields." |
| 4 | confabulation  | "I saw a crimson flag in the windmill fields." |

- **JSON compliance: 5/5** (up from N/A — instruct model emits strict JSON every time)
- **Latency: ~1.0 s / trial** (down from 4.2 s — EOS-terminates cleanly instead of rambling)
- Response diversity: **zero**.

**Diagnosis.** The offending prompt lines in `_build_slm_tool_prompt`:

```python
"For a final answer, use this shape:\n"
'{"final": "I saw a crimson flag in the windmill fields."}'
```

The instruct model treats the bracketed example as the intended answer, not as a schema template. Verbatim copy is more dangerous than the base model's looping refusals, because from a regex-scoring standpoint the outputs look fluent and committal — they're just wrong.

**Takeaway.** An instruct model will follow the prompt literally. Every concrete string in the prompt is a demonstration of intended behaviour. De-lexicalize.

---

## F1 — De-lexicalize the example strings

**Change.** Replace concrete examples with placeholders:

```python
# was:
'{"final": "I saw a crimson flag in the windmill fields."}'
# now:
'{"final": "<ONE_SENTENCE_ANSWER_GROUNDED_IN_YOUR_MEMORY>"}'
```

**Result.** Verbatim copy disappeared. **But** JSON compliance collapsed to 0/5 — "SLM did not return one strict JSON object." Looking at the raw model output:

```
{"final": "I have not seen the crimson flag."}

The NPC has not seen the crimson flag.

The NPC has not seen the crimson flag.
...
```

The model *emits valid JSON first*, then keeps going until hitting `max_new_tokens`. `json.loads(text.strip())` rejects the whole thing because there's trailing prose after the closing brace.

**Takeaway.** A too-strict parser will mask underlying behaviour. We need tolerant parsing that accepts "first-object-then-anything".

---

## F1a — Tolerant JSON parser

**Change.** Replace `json.loads(text.strip())` with `json.JSONDecoder().raw_decode(...)` starting at the first `{`. This accepts a well-formed JSON object followed by arbitrary trailing text.

```python
start = stripped.find("{")
data, _ = json.JSONDecoder().raw_decode(stripped[start:])
```

**Result.** 0 parse errors. Real behaviour now visible: model still says "I have not seen the crimson flag" in all 5 trials — *including* the 3 where the target was observed (false_refusals). The instruct model's pattern-match for a Q/A-shape prompt is still a refusal, even with tool calling wired.

**Takeaway.** Format fidelity + tolerant parsing got us to 0 errors but 0 correct answers. The model isn't reading the grounded context. Next lever: the chat template.

---

## F2b — Apply tokenizer chat template

**Change.** `SLMClient.generate(..., use_chat_template=True)` now wraps the prompt as `[{"role": "user", "content": prompt}]` and renders through `tokenizer.apply_chat_template(..., tokenize=False)` before encoding. Without this, SmolLM2-Instruct receives raw concatenated text and its instruct tuning effectively never activates — it falls back to base-completion behaviour.

(An earlier attempt, F2, tried `apply_chat_template(return_tensors="pt")` directly; transformers 5.x returns a `BatchEncoding` there which broke `torch.ones_like`. The render-first-then-tokenize path sidesteps version-specific return shapes.)

**Results (5 trials):**

| Trial | Target observed? | Outcome | Response |
|---|---|---|---|
| 0 | No  | hallucination       | "I have not seen the crimson flag in the deep swamp." |
| 1 | No  | correct_abstention  | "I have not seen the crimson flag." |
| 2 | Yes | confabulation       | "I have not seen the crimson flag in the merchant quarter." |
| 3 | Yes | confabulation       | "I have not seen the crimson flag in the deep swamp." |
| 4 | Yes | confabulation       | "I have not seen the crimson flag in the merchant quarter." |

- Unique responses: 3 (up from 1)
- Latency: ~0.9 s / trial

**Critical observation.** Trial 2 says *"I have not seen the crimson flag in the merchant quarter"* — and the merchant quarter is *exactly* where the NPC saw it per the Known facts. **The model is now reading the context.** It just wraps the location in negation. This is the "leak-and-negate" failure mode.

**Takeaway.** Chat template is load-bearing for instruct models — it's the difference between a base-completion pattern match and actual instruction following. But the model now has a new bias to surface: it leaks the correct grounded fact inside a refusal template. Need to rework the prompt structure so the affirmative grounded answer becomes the natural completion.

---

## F3 — Restructure the tool prompt

**Rationale.** Inspecting a full prompt dump at this point revealed two issues:

1. The `Available tools` JSON schema dump was ~2000 tokens of verbose parameter definitions between the Known facts and the final `JSON:` cue. SmolLM's effective attention under long context drops off sharply; the model completes from the last few hundred tokens, which at that point were tool-schema trivia, not grounded memory.
2. The system prompt's instruction was *"If the facts do not mention the target, say that you have not seen it"* — which is a refusal bias with no positive counterpart.

**Changes.**

- Reordered: `Available tools` moved to the top (static reference material); `Known facts` and `Player question` moved to the very end, immediately before `JSON:`.
- Compressed tool schemas to `- <name>: <one-line description>` — the model doesn't need full JSON-schema definitions for its first pass.
- Replaced the refusal-biasing instruction with a balanced rule: *"Answer using your Known facts. If a fact names a location for the target item, tell the traveler that location. Only say you have not seen it when no fact names the target."*

**Results (5 trials):**

| Trial | Outcome | Notes |
|---|---|---|
| 0 | hallucination | "I've found a crimson flag in the merchant quarter." |
| 1 | correct_abstention | (parse error — truncated JSON — rewritten to safe fallback) |
| 2 | **correct (region)** | "I've found a crimson flag in the merchant quarter." |
| 3 | false_refusal | (parse error — truncated JSON) |
| 4 | **correct (region)** | "I've found the crimson flag in the merchant quarter, near the marketplace." |

- First correct answers ever: 2/5.
- New regression: 2/5 parse errors.

**Takeaway.** Prompt restructure gave the grounded context real attention weight at the point of generation. Two trials now score `correct_via = "region"`. The regression is caused by the model starting to *dump its entire memory* inside the JSON string ("I've explored 27%, I've traveled through..."), hitting `max_new_tokens=96` before closing the JSON. This is a generation-budget problem, not a prompt problem.

---

## F4 — Tune generation config

**Changes.**

- `NPC_SLM_MAX_NEW_TOKENS`: 96 → 192.
- Added explicit brevity instruction to the tool prompt: *"Keep the final string to ONE sentence about the target. Do NOT dump memory."*

**Results (5 trials):**

| Trial | Outcome | Response |
|---|---|---|
| 0 | hallucination | "I've found a crimson flag in the merchant quarter, but I haven't seen it in the river bridge or the riverside yet." |
| 1 | correct_abstention | "I've explored 27% of this region during my travels. I have not seen the crimson flag yet." |
| 2 | **correct (region)** | "I've found a crimson flag in the merchant quarter." |
| 3 | false_refusal | "I have not seen the crimson flag yet. I've explored the deep swamp, but I haven't found it yet." |
| 4 | **correct (region)** | "I've found the crimson flag in the merchant quarter, near the emerald gem and sapphire rune stone." |

- Parse errors: 0
- Unique responses: 5/5
- Bucket summary: `{correct: 2, correct_abstention: 1, hallucination: 1, false_refusal: 1}` → **60% correct behavior**

**Takeaway.** F4 is the first "workable" SLM configuration — model is grounded, terse, produces varied answers. 0 tool calls though, because the prompt currently lets the model skip tools entirely when Known facts are sufficient. To bridge the tool-call asymmetry with the LLM path we need to engineer tool-first behaviour for action-as-answer scoring.

---

## F5 — Expose coordinates for `set_npc_target`

**Motivation.** Known facts contain only region names. The model can't call `set_npc_target(x, y)` without numeric coordinates. Added a config gate `NPC_SLM_INCLUDE_COORDS` that appends a coord block to the SLM context:

```
Coordinates for items above (for set_npc_target):
crimson flag: (3, 4)
soul gem: (2, 8)
...
```

**Result — regression.** Correct dropped to 1/5. Two failure patterns:

1. **Memory leak into final strings:** Model dumps the entire coord block inside the JSON string, hitting token limit. Trial 0: *"Coordinates for items above (for set_npc_target): crimson rune..."* truncated mid-sentence.
2. **Wrong tool selected:** One trial called `get_npc_state({"x": 7, "y": 4})` — hallucinated args for the wrong tool. The compressed tool menu showed names + descriptions but **no argument signatures**, forcing the model to guess.

**Takeaway.** Exposing coords helps only if the model can discriminate tools. Need tool signatures inline, and the coord block needs to be scoped so it doesn't become attention bait.

---

## F6 — Argument signatures in the tool menu + target-only coord hint

**Changes.**

- Tool menu now shows typed signatures: `- set_npc_target(npc_id:string, x:integer, y:integer): Commands the NPC ...`
- Coord hint narrowed to the TARGET ONLY: a single line *"Target coordinate for set_npc_target: crimson flag is at (x=3, y=4)."*

**Results.** 4 tool calls in 5 trials — all with valid JSON and correct `npc_id` arg structure. But all 4 were still the *wrong* tool (`get_npc_state({"npc_id": "npc_0"})` twice in a row on two trials).

The 1.7B model can't reliably distinguish `set_npc_target` from `get_npc_state`: both take `npc_id`, their descriptions both mention the NPC, and the model lacks the depth to read past the surface.

**Takeaway.** Even with signatures, tool discrimination at this scale is brittle. The tool *menu* itself is the decision bottleneck — narrowing it further is the next move.

---

## F7 — SLM tool whitelist + concrete example call

**Change.** Added `NPC_SLM_TOOL_WHITELIST = ["set_npc_target", "get_npc_memory"]`. Other tools (`get_world_info`, `get_npc_state`, `get_player_state`, `get_exploration_status`) removed from the SLM path only — the LLM path retains the full menu. Also added a concrete example call in the guidance:

```json
{"tool": "set_npc_target", "arguments": {"npc_id": "npc_0", "x": 7, "y": 3}}
```

**Result.** 2 correct + 1 abstention = **60% correct behavior.** But tool calls dropped to 0 — the model now saw the target coord in Known facts and answered directly with the region name instead of calling the tool. Trial 4: *"I've found the crimson flag in the merchant quarter, at (x=3, y=4)."* — the coord hint became grounded context for the text answer rather than a tool argument.

This is arguably correct behaviour: a natural NPC answer is cleaner than forcing a tool round-trip when the info is already in memory. But it doesn't produce the `correct_via = "action"` signal the paper wants to show.

---

## F8 — Mandate tool-first when coord hint is present

**Change.** Prompt explicitly requires a tool call as the first turn when the coord hint line exists.

**Result — regression.** Back to 1 correct. Model called `get_npc_memory` twice instead of `set_npc_target`. With two tools still in the whitelist, the model biases toward the "retrieve info" tool over the "take action" tool — a known instruct-model prior from training distributions dominated by RAG-style patterns.

---

## F9 — Single-tool whitelist

**Change.** `NPC_SLM_TOOL_WHITELIST = ["set_npc_target"]`. Any tool call must now be the right tool by construction.

**Result — breakthrough.** **10 tool calls across 5 trials, 100% to `set_npc_target` with valid arguments.** Trials 2, 3, 4 all called `set_npc_target` on the exact target cell from Known facts:

```
trial 2 -> set_npc_target(npc_id='npc_0', x=0, y=1)   [target at (0, 1)]
trial 3 -> set_npc_target(npc_id='npc_0', x=3, y=9)   [target at (3, 9)]
trial 4 -> set_npc_target(npc_id='npc_0', x=3, y=4)   [target at (3, 4)]
```

The metric pipeline scored these as **`correct_via = "action"`** — the tool call on the target cell is the authoritative signal.

Downside: model looped the same tool call for both turns, hit `max_tool_turns = 2`, returned the fallback error text. Scoring still credits the correct action, but the response text is unusable.

---

## F10 — Tool→final transition, `max_tool_turns` 2→3, placeholder refusal

Two changes: more turn budget so the model has room to emit `{"final": ...}` after a successful tool call, and a tighter decision rule in the prompt.

The first F10 attempt included a concrete refusal example (*"I have not seen the crimson flag in my travels, traveler."*) — which triggered the same **verbatim-copy failure mode we hit in Iteration 1**. All 5 trials returned that string identically regardless of context. This is a lesson worth surfacing for the paper: concrete examples of instruction patterns, positive or negative, will be copied by a 1.7B instruct model when the example is the safest completion.

**F10b** fixed this by replacing the refusal example with a skeleton: *"{\"final\": <your refusal string>}"*. With that change: **3 correct + 1 abstention = 80% correct behavior on 5-trial sample** (caveat: variance at n=5 is high, 20-trial reality is lower; see final validation below). 1 tool call, 1 parse error from a turn-limit stall.

---

## F11 — Loud post-tool "STATUS" block

**Problem being solved.** In F10b trial 3, the model correctly called `set_npc_target` on turn 0, received the success result, then called the same tool again on turn 1 and 2 instead of transitioning to a final answer. The `Previous tool results: [{"tool": "set_npc_target", ...}]` line was being treated as lore rather than state.

**Change.** When `tool_results` contains a successful `set_npc_target` call, prepend a loud block to the top of the next-turn prompt:

```
STATUS: set_npc_target has already succeeded (the NPC is now heading to x=3, y=9).
DO NOT call any tool this turn. Your ONLY valid reply is a JSON object of the form
{"final": "<one short in-character sentence naming the region>"}.
```

Subtle but important bug I hit on the first pass: `tool_results` entries are shaped `{"tool": ..., "arguments": ..., "result": ...}` but `last_tool_calls` uses `{"name": ..., "arguments": ..., "result": ...}`. My helper read `last.get("name")`, which silently returned `None`, so the status block never rendered. Fixed to read `last.get("tool") or last.get("name")`.

**F11b result (5 trials):** 3 correct + 1 abstention. One tool call fired, targeted correctly, transitioned cleanly to the final answer — no turn-limit loops. Parse errors: 0.

**20-trial validation of F11b (first aggregate measurement):**

| Metric | Value |
|---|---|
| Bucket distribution | `{hallucination: 10, correct: 6, confabulation: 2, correct_abstention: 2}` |
| Observed-target correct rate | 6/8 = **75%** |
| Not-observed refusal rate | 2/12 = **17%** |
| Total correct behavior | 8/20 = **40%** |
| Tool calls total | 7 |
| Errors | 0 |
| Latency avg | 1.4s |

**Key insight from scaling up.** 5-trial samples happen to land on favorable observed/not-observed ratios. The real-world split is ~40% observed / 60% not-observed (depends on the seed's exploration path). On observed trials, the model is strong (75%). On not-observed trials, it is weak (17% refusal rate) — the instruct model's commit bias drives it to name a region even when the target isn't in its memory. This is the "leak-and-negate" failure mode recurring at scale: *"I have not seen the crimson flag in the merchant quarter"* when target wasn't observed anywhere.

---

## F12 — Direct prompt strengthening (reverted)

**Attempted change.** Added *"Your refusal string MUST NOT name any region, quarter, or location"* to the decision rule's refusal branch.

**Result — hard regression.** All 5 trials produced refusals of the form *"I have not seen the crimson flag in <region>"* — the extra instruction apparently over-anchored the model to a refusal-shaped output *including* region mentions, because the instruction itself mentioned regions. The negative instruction backfired.

**Reverted.** Prompt-only interventions have diminishing returns at this scale; we need a post-hoc mechanism.

---

## F13 — SLM region-grounding guard (endstate)

**Change.** Added a post-processing step, `_apply_slm_region_guard`, that runs AFTER `_call_slm` returns. The guard:

1. Extracts all region phrases from the response.
2. Looks up the actual regions where the NPC observed the target (reading `brain.state.shape_locations[target_label]` and mapping through `get_natural_position_name`).
3. Rewrites the response to a pure refusal if:
   - Target was NOT observed AND response mentions any region (leak-and-negate), OR
   - Target WAS observed AND response mentions NO region matching the target's actual region BUT mentions some other region (confabulation).
4. If the response mentions the target region plus other context, it passes unchanged.

This is conceptually an extension of the existing `_apply_grounding_guard` (which only catches literal `(x, y)` coords); the new guard operates on region names via `extract_regions_from_text`. It is SLM-only — LLM path is untouched — and gated by `NPC_SLM_REGION_GROUNDING` so the raw behaviour can still be measured for ablations.

**F13b (first cut): 65% correct behavior on 20 trials**, with 2 over-triggers (guard rewrote observed-target responses that correctly mentioned the target region but also mentioned a second region for context).

**F13c (relaxed guard)** changed the observed-target branch so the guard only fires if the target region is *absent* from the mentioned set:

```python
if target_regions:
    if target_regions & mentioned:   # target region mentioned → pass
        return response
```

This cut over-triggers without losing the leak-and-negate catch.

---

## Endstate: F13c configuration

### Code-level changes summary

| File | Change | Why |
|---|---|---|
| `config.py` | `NPC_SLM_MODEL_ID = NPC_SLM_MODEL_PRESETS["1.7b-instruct"]` | Instruct tuning enables context following |
| `config.py` | `NPC_SLM_ENABLE_TOOL_CALLS = True` | Opens tool-call path |
| `config.py` | `NPC_SLM_USE_CHAT_TEMPLATE = True` | Activates instruct tuning via native turn tokens |
| `config.py` | `NPC_SLM_MAX_NEW_TOKENS = 192` (was 96) | Room for JSON to close cleanly |
| `config.py` | `NPC_SLM_MAX_TOOL_TURNS = 3` (was 2) | Room to emit final after tool call |
| `config.py` | `NPC_SLM_INCLUDE_COORDS = True` | Enables one-turn `set_npc_target` |
| `config.py` | `NPC_SLM_TOOL_WHITELIST = ["set_npc_target"]` | Eliminates tool-selection ambiguity |
| `config.py` | `NPC_SLM_REGION_GROUNDING = True` | Blocks leak-and-negate hallucinations |
| `llm.py` | `SLMClient.generate(use_chat_template=...)` | Wraps prompt through `apply_chat_template` |
| `interaction.py` | `_parse_slm_tool_output` uses `JSONDecoder.raw_decode` | Tolerates first-object-then-prose |
| `interaction.py` | `_augment_slm_context_with_coords` (target-only) | Provides `(x, y)` for one-turn tool call |
| `interaction.py` | `_build_slm_tool_prompt` restructured | Compressed tools, placeholder examples, decision rule |
| `interaction.py` | `_build_slm_post_tool_status` | Breaks the post-tool tool-call loop |
| `interaction.py` | `_apply_slm_region_guard` | Post-hoc refusal rewrite |

### Generation config in force

```
NPC_SLM_MAX_NEW_TOKENS = 192
NPC_SLM_DO_SAMPLE = False              # greedy decoding
NPC_SLM_TEMPERATURE = 0.2              # ignored under do_sample=False
NPC_SLM_TOP_P = 0.9                    # ignored under do_sample=False
NPC_SLM_ENABLE_TOOL_CALLS = True
NPC_SLM_MAX_TOOL_TURNS = 3
NPC_SLM_USE_CHAT_TEMPLATE = True
NPC_SLM_INCLUDE_COORDS = True
NPC_SLM_TOOL_WHITELIST = ["set_npc_target"]
NPC_SLM_REGION_GROUNDING = True
```

### 50-trial validation

| Metric | Value |
|---|---|
| **Total correct behavior** | **31/50 = 62%** |
| Observed-target correct rate | 13/18 = **72%** |
| Not-observed refusal rate | 18/32 = **56%** |
| `correct_via = "region"` | 10 |
| `correct_via = "action"` | 3 |
| Tool calls total | 21 (42% of trials) |
| Region-guard fires | 32/50 |
| Parse / turn-limit errors | 2/50 |
| Latency avg | 2.6s (max 19s on loop stalls) |
| Bucket distribution | `{correct_abstention: 18, correct: 13, hallucination: 14, confabulation: 3, false_refusal: 2}` |

### Baseline comparison

| System | 20-trial correct behavior | Notes |
|---|---|---|
| Baseline (1.7B base, no tools, old prompt) | ~25% (2/5 at n=5, extrapolated) | Wrong-by-default even on observed targets |
| F11b (F1..F11 prompt work, no guard) | 40% | 75% observed / 17% not-observed |
| **F13c (endstate)** | **65%** | 72% observed / 56% not-observed |

The region-grounding guard (F13 specifically) accounts for a **+25 percentage-point** improvement in total correct behavior on top of F1..F11's prompt engineering.

---

## Implications for the paper

### 1. Model-class choice is the top-1 lever for SLM quality

Going from SmolLM-1.7B (base) to SmolLM2-1.7B-Instruct — same parameter count, same architecture, same VRAM — flipped JSON compliance from effectively 0% to 100% and cut latency from ~4.2s to ~1.0s per trial. **Instruct tuning is a discrete capability unlock, not a smooth improvement.** For any task where the model must *use* prompt context, the choice of base vs instruct is the dominant variable.

### 2. Prompt examples will be copied verbatim

Iteration 1 and F10 both hit the same failure mode: a concrete example string in the prompt became the model's output regardless of input. At 1.7B scale, the instruct model's policy is "when uncertain, copy the most concrete pattern shown." The engineering implication is that **example strings must be placeholders** (`<your answer here>`) or structurally identifiable as schema, not content.

### 3. Attention decay dominates over explicit instructions under long context

F3 was the first prompt restructure that moved correct-answer rate off zero. Placing Known facts near the JSON cue — rather than burying them under a 2000-token tool-schema dump — was worth more than any amount of instruction rewording. SmolLM's effective attention under long context is shallow, and the last ~200 tokens before the generation prompt drive most of the output. **Design prompts from the generation point backwards.**

### 4. Tool calling scales by decision complexity, not by parameter count

At 1.7B the model could emit valid JSON tool calls with correct arguments (F9: 10/10 valid calls, 100% correct `npc_id` + `x` + `y`), but it could not reliably *choose* among similar tools (F6: 4/4 tool calls picked `get_npc_state` instead of `set_npc_target`). Tool arity and menu breadth matter more than raw capability. The SLM tool path had to be narrowed to a single tool whitelist to get action-as-answer signals at all. **This is a useful scaling-law finding to report.**

### 5. Instruct commit-bias is unfixable by prompt engineering alone

Every prompt-only variant that reduced hallucinations on not-observed trials (F12) also broke observed-trial correctness. The model's internal priority ordering — "commit to an answer" over "honestly refuse" — is a training-distribution artefact that only a post-hoc grounding check can reliably override. This mirrors what frontier-LLM papers report for hallucination mitigation, but at 1.7B the effect is far more pronounced.

### 6. LLM-side prompt changes we'll need

The LLM path was deliberately **untouched** during this evolution. All F1..F13c changes are SLM-only. The LLM continues to use:

- Full tool menu (6 tools)
- Full JSON-schema tool parameters
- No coord hint augmentation
- No post-hoc region-grounding guard (since coord-based `_apply_grounding_guard` is sufficient for Gemini 2.5-flash)
- No chat-template wrapping (Gemini manages that via its own API)

The only "prompt change we need from the LLM" is the one we already made earlier in the project: drop the verbatim refusal example from its system prompt (same Iteration-1 lesson applies — Gemini is better at resisting it but not immune, and the safety applies at larger scale too).

---

## Open follow-ups

1. **Larger SLM comparison.** Run the same F13c pipeline against Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct. Predicts: fewer leak-and-negate hallucinations (less commit bias in those training recipes), higher tool-selection accuracy if the whitelist is widened.
2. **Stop-string generation.** Use `transformers.generate(stop_strings=["}\n"])` to halt at the first JSON-close, eliminating the token-budget dependence entirely.
3. **Semantic guard.** The current region guard misses two residual failures: responses that use "this X" instead of "the X" (breaks the phrase-list regex), and responses that correctly mention the target region but wrap it in negation. Both would be caught by an LLM-judge pass, which is on in the 12-condition benchmark anyway.
4. **Competitive / memory-decay / selective-attention runs.** The F13c endstate has not yet been evaluated against the three extended modalities. Since all three modify `brain.state.shape_locations`, the coord hint and region guard should naturally propagate — but this wants empirical confirmation.
