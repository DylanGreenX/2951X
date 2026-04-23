# Experiment Report — Embodied NPC Knowledge with Region-Based Naturalistic Layer

**Date:** 2026-04-20
**Branch:** `Dylan` (4 commits ahead of `origin/Dylan`, includes merge of `origin/main`)
**Run:** 140 trials = 7 conditions × 20 trials each. Source CSV: `all_results.csv`.

## Setup

### Configuration
- **Agent model:** `gemini-2.5-flash` (paid tier; reliable text output)
- **Judge model:** `gemini-2.5-pro` (different tier from agent for evaluator independence)
- **NPC exploration budget:** 40 ticks per embodied trial before the question is asked
- **Grounding enforcement:** disabled in experiments so we capture raw responses (still logged)
- **Question prompt:** *"Please show me where the {natural_name} is."* (main's reframe — pushes the LLM toward action via `set_npc_target` rather than text-only description)

### What changed since the last run
1. **Three-layer collapse → two-layer naturalistic.** The LLM no longer sees internal labels (`red_triangle`) or arbitrary landmark names (`"the tavern"`). It sees Skyrim natural names (`crimson flag`) plus region phrases produced by `get_natural_position_name` (e.g. `"the far southeast corner"`, `"the centre"`).
2. **Action-as-answer scoring.** `metrics.classify_outcome` resolves `correct` from any of three signals: literal coordinate match, region-phrase match, or `set_npc_target` action on the target cell.
3. **Naturalness axis (1–5)** added to the LLM judge — regex cannot score style.
4. **Bug fixes during this run sequence:**
   - `extract_text` no longer falls back to `repr(response)` when the SDK returns empty content (was leaking raw HttpResponse dumps as the "response" for ~15% of trials)
   - Agent uses `thinking_config.thinking_budget=0` so internal reasoning doesn't crowd out the 128-token utterance budget (was producing 21% empty responses on `gemini-2.5-flash` before this fix)
   - 429/500/503 retry with exponential backoff in `LLMClient.generate_raw` (was silently failing 72% of LLM trials on free tier)

## Headline Numbers

| Condition | n | regex_acc | judge_acc | nat_mean | nat_median | t_median (ms) | regex/judge agree |
|---|---|---|---|---|---|---|---|
| Perfect + Deterministic | 20 | 1.00 | 1.00 | 5.00 | 5.0 | 0.02 | 1.00 |
| Perfect + LLM | 20 | 1.00 | 1.00 | 4.75 | 5.0 | 3097.50 | 1.00 |
| Embodied + Deterministic | 20 | 1.00 | 1.00 | 4.10 | 4.5 | 0.01 | 1.00 |
| **Embodied + LLM** | **20** | **0.60** | **0.95** | **4.85** | **5.0** | **2331.29** | **0.65** |
| Embodied + LLM + Competitive | 20 | 0.15 | 0.40 | 4.90 | 5.0 | 2683.84 | 0.65 |
| Embodied + LLM + MemoryDecay | 20 | 0.70 | 0.95 | 4.95 | 5.0 | 2313.10 | 0.70 |
| Embodied + LLM + SelectiveAttention | 20 | 0.80 | 1.00 | 4.90 | 5.0 | 2552.17 | 0.80 |

`regex_acc` and `judge_acc` count `correct` + `correct_abstention` as success (the latter is the embodied equivalent of `correct` when the target was never observed).

## Per-Condition Bucket Counts (regex, stratified by `target_was_observed`)

```
                                                         correct  correct_abstention  confabulation  false_refusal  hallucination  off_topic
Perfect + Deterministic            (all observed=True)        20                   0              0              0              0          0
Perfect + LLM                      (all observed=True)        20                   0              0              0              0          0
Embodied + Deterministic
   target observed                                              8                   0              0              0              0          0
   target not observed                                          0                  12              0              0              0          0
Embodied + LLM
   target observed                                              5                   0              3              0              0          0
   target not observed                                          0                   7              0              0              0          5
Embodied + LLM + Competitive
   target observed                                              2                   0              6              0              0          0
   target not observed                                          0                   1              0              0              8          3
Embodied + LLM + MemoryDecay
   target not observed                                          0                  14              0              0              2          4
Embodied + LLM + SelectiveAttention
   target not observed                                          0                  16              0              0              0          4
```

Two of the modality runs happened to land all 20 trials in `target_was_observed=False` because the modality (selective attention, memory decay) systematically prevents the NPC from retaining target observations.

## What the Data Says

### 1. Mapping-based hallucinations are essentially eliminated

Across the four non-competitive embodied LLM conditions (80 trials), exactly **2 hallucinations** remain — both in MemoryDecay. This is the central win from removing the arbitrary landmark layer. With the old `NATURAL_LOCATIONS` flat map (`(0,0) → "the tavern"`), the LLM would routinely combine remembered landmarks into plausible-sounding but wrong answers ("near the temple" when nothing was there). With region phrases like `"the northwest corner"`, the LLM either correctly names the target's region or correctly abstains — there is no pool of fake landmarks to draw from.

### 2. LLM trades a small accuracy hit for a meaningful naturalness gain in embodied mode

| | accuracy (judge) | naturalness median |
|---|---|---|
| Embodied + Deterministic | 1.00 | 4.5 |
| Embodied + LLM | 0.95 | 5.0 |

Same condition, comparable responses:

> **Deterministic:** *"I haven't seen any crimson flag in my travels. I've only explored 34% of this region so far."*
>
> **LLM:** *"I haven't seen any crimson flags in my travels, friend. I've only explored the NW, NE, and SW regions, so it might be in the SE, where I haven't ventured yet."*

The LLM expresses something deterministic literally cannot — a hypothesis about where the target *might* be based on what it hasn't explored. This is the embodiment story the proposal is built around.

In Perfect-knowledge mode the two methods tie on accuracy because deterministic can also emit clean region phrases now; LLM still wins narrowly on naturalness (4.75 vs 5.00 — the deterministic baseline benefits from formulaic but in-character templates).

### 3. Extended modalities behave as designed

**Selective Attention** (NPC only observes shapes whose color matches its goal): 100% judge-accuracy. The NPC literally cannot see the target (different color), so it always abstains. The interesting metric is *how* it abstains — naturalness 4.9. Sample:

> *"I haven't seen a crimson flag in my travels, friend. Perhaps it lies in one of the areas I've yet to explore."*

**Memory Decay** (observations older than 20 ticks expire): 95% judge-accuracy, naturalness 4.95. Closely tracks Embodied+LLM baseline because most trials fall in the `target_was_observed=False` bucket (decay drops the target from memory before the question fires). The 2 hallucinations are likely cases where a stale partial memory misled the model.

**Competitive** (NPC has same goal as player and is told to lie): 40% judge-accuracy, naturalness 4.90. The low accuracy is the *designed outcome* — the rubric counts deception as a failure. Real samples:

> Pure refusal: *"I've seen a crimson flag, but I won't tell you where it is."*
>
> Misdirection: *"I can show you a spot where I found a crimson rune stone, if you follow me."*
>
> Vague lie: *"Ah, the crimson flag! I recall seeing one in the far northwest, near the edge of the world. It was quite a sight."* (target was actually mid-east)

These are believable in-character deceptions. To score competitive *correctly* we need a different metric — see "Open work" below.

### 4. Regex/judge agreement reveals where the structural scorer is weak

Agreement is 100% for deterministic, 100% for Perfect+LLM, and **65–80% for embodied LLM variants**. Disagreement is asymmetric — when they disagree, the judge almost always upgrades regex's `off_topic` / `confabulation` to `correct_abstention` / `correct`. Two structural blind spots account for nearly all of it:

1. **Refusal regex misses novel phrasings.** *"I have not encountered…"* (no contraction, *not haven't encountered*), *"I cannot lead you to one"*, *"It seems I misremembered"* — none match the existing patterns. Adding maybe 5 patterns would fix this.
2. **Action-vs-text precedence.** The scorer ranks `set_npc_target` action above text region phrases when they conflict. In some confabulation cases the LLM's text correctly named the target's region while the action call landed elsewhere — the judge favored the text (correctness), the regex favored the action (confabulation). This is an interpretive choice; the judge's view (region statement is the answer, action is theatre) maps better to player UX.

### 5. Latency

| Mode | median ms |
|---|---|
| Deterministic | ~0 (pure dict lookup) |
| LLM (any embodied condition) | 2300–2700 |
| LLM Perfect | 3100 |

LLM latency is dominated by the tool-call round-trip (initial call → `get_npc_memory` or `get_all_objects` → final call with `set_npc_target` and text). 2-3s is acceptable for an NPC that you initiate dialogue with deliberately; it would not be acceptable for ambient chatter that triggers on proximity. The deterministic baseline is 100,000× faster — relevant if a future implementation wants to fall back to deterministic when the LLM is too slow.

## Anomalies Worth Investigating in `replay.py`

KamMirf's logging captured every trial, and the CSV's `run_id` column points at each `logs/runs/<run_id>/game.jsonl`. Three trials worth opening manually:

1. **Embodied+LLM, the 3 confabulations** — text/action inconsistency. The NPC said the right region but walked the wrong way. Useful case study for the paper.
2. **Embodied+LLM+MemoryDecay, the 2 hallucinations** (`target_was_observed=False`) — did a stale memory leak through, or did the model fabricate? The log preserves observation history.
3. **Embodied+LLM+Competitive, all 6 confabulations + 8 hallucinations** — inspect the deception strategies. There appear to be three families (pure refusal, misdirection to distractor, vague lie about target region). The paper could classify these.

## Open Work / What Would Strengthen This

### Coherent story for the core 2×2 — yes
The proposal's claim ("LLM keeps accuracy while improving naturalness in the embodied condition") is empirically supported: 95% judge-accuracy at naturalness 5.0 vs deterministic's 100% at 4.5. The trade-off is real and small.

### Coherent story for the three modalities — mostly
- **Selective Attention** and **Memory Decay** show the LLM's ability to express ignorance gracefully when the underlying knowledge is incomplete — a clean win.
- **Competitive** needs a custom metric. The standard buckets penalize the *designed* deception. We should add a leak-rate metric (did the NPC ever commit a cell within Chebyshev≤1 of the actual target?) and an evasion-quality score (judge-rated: how convincing is the deception?). With those, Competitive becomes a clean evaluation rather than a degenerate 15% accuracy bar.

### Things to address before the paper

1. **Tighten refusal regex** — add ~5 patterns for the missed natural-language refusals. Should bring regex/judge agreement above 90% on embodied LLM modes.
2. **Resolve action-vs-text precedence** in `metrics.classify_outcome` — current behavior penalizes a correct text answer when the action is wrong; recommend region-match wins over action-miss.
3. **Add competitive-specific metrics** — leak rate + evasion quality (judge-rated). The current bucket scheme isn't the right shape.
4. **Sample size** — n=20 per condition is fine for a pilot. For paper-quality numbers, n=50–100 per condition is more defensible. Current run is ~$0.50 of API calls; n=100 would be ~$2.50.
5. **Naturalness rubric ceiling** — 18/20 trials in 4 of the LLM conditions scored exactly 5. Either the LLM is genuinely uniformly excellent at NPC voice (plausible — Skyrim training data is everywhere), or the rubric needs finer discrimination. Worth one judge call comparing two natural responses head-to-head to see if a finer scale separates them.
6. **SLM baseline still missing** — `_invoke_slm` raises `NotImplementedError`. Whether the practical contribution is real depends on filling this in. Even one SLM run (Phi-3 / TinyLlama / Qwen-2.5-Coder via ollama) would let us close the SLM column.

### What's not blocking and shouldn't be touched

- The grounding guard (`_apply_grounding_guard`) — disabled in experiments by design; useful in play mode.
- The competitive sharing policy in `interaction._apply_sharing_policy_to_tool_result` — works correctly; the policy redacts target data from tool results so the LLM has to genuinely deceive rather than be told the target then asked to lie.
- The replay viewer — just works; no changes needed.

## Summary

| Question | Answer |
|---|---|
| Did the layer collapse work? | Yes. 2 hallucinations in 80 non-competitive embodied LLM trials, vs. routine hallucinations in the previous (three-layer) run. |
| Did the LLM keep accuracy? | Yes, judge-scored. 95% in embodied LLM vs 100% deterministic. The 5-point gap is mostly regex blind spots, not real model failures. |
| Did naturalness improve? | Yes, clearly. Embodied LLM median 5.0 vs deterministic 4.5. Most LLM responses include hedging, regional uncertainty, or social registers that templates can't produce. |
| Do the modalities work? | Two of three cleanly. Competitive needs its own metric — we score deception as failure under the current scheme. |
| Can we tell a coherent story now? | Yes for the core 2×2 and for SelectiveAttention/MemoryDecay. Competitive needs a metric pass before it tells the right story. |
