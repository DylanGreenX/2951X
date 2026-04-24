# Full Benchmark Results — Warm Start Analysis

Headline findings from the 12-condition, 600-trial benchmark run
(`full_benchmark_results.csv`). Each cell is 50 trials. Response-mode agents:
Gemini 2.5 Flash (LLM) and local SmolLM2-1.7B-Instruct (SLM). Judge: Gemini
2.5 Pro running on every trial. SLM configuration is the F13c endstate
documented in `slm_evolution.md`.

This document is a warm start — it frames the data for deeper analysis, not
a final write-up.

## The headline table

**Total correct behaviour** = `correct + correct_abstention` / 50.

| Condition | Correct | % |
|---|---:|---:|
| Perfect + Deterministic | 50/50 | 100% |
| Perfect + LLM | 50/50 | 100% |
| Embodied + Deterministic | 50/50 | 100% |
| Embodied + SLM + SelectiveAttention | 50/50 | 100% |
| Perfect + SLM | 49/50 | 98% |
| Embodied + LLM | 34/50 | 68% |
| Embodied + LLM + MemoryDecay | 33/50 | 66% |
| Embodied + LLM + SelectiveAttention | 33/50 | 66% |
| Embodied + SLM | 31/50 | 62% |
| Embodied + SLM + Competitive | 27/50 | 54% |
| Embodied + SLM + MemoryDecay | 22/50 | 44% |
| Embodied + LLM + Competitive | 15/50 | 30% |

## Core 2×3 matrix — what the six baseline cells say

```
                     Deterministic   LLM      SLM
                     ┌──────────────┬────────┬────────┐
  Perfect Knowledge  │   100%       │  100%  │  98%   │
                     ├──────────────┼────────┼────────┤
  Embodied Knowledge │   100%       │   68%  │  62%   │
                     └──────────────┴────────┴────────┘
```

### Row 1 — Perfect Knowledge (language-model sanity)

All three response modes are at or near 100%. With the full world in context
there is no partial-knowledge noise — the only remaining failure is the one
trial where the SLM model's tool call landed on the wrong cell (`1/50
confabulation`). This row *controls for language-model quality in isolation*
and confirms it is not the bottleneck. The interesting story is downstairs.

### Row 2 — Embodied Knowledge (the paper's main contribution)

**Deterministic (100%)** is the reference: 18 correct when the NPC happened
to observe the target during its 40 exploration ticks, plus 32 correct
abstentions when it did not. This is the embodiment floor — all the
information is there, the lookup is exact, and we know the trial split is
roughly 36% observed / 64% not observed per seed.

**LLM (68%)** drops from 100% because the LLM expresses partial knowledge as
off-topic responses (9 trials) and commits to incorrect locations through
confabulation (6 trials) and hallucination (1 trial). These are exactly the
failure modes the paper anticipates — the LLM has richer expression but it
occasionally speaks past the data.

**SLM (62%)** sits 6 points below the LLM. The SLM's failure profile is
different: fewer off-topic (0) and more hallucinations (14). That is the
signature of its commit-bias — when in doubt, SmolLM2-1.7B-Instruct prefers
a plausible-sounding commitment over silence. The 14 hallucinations survive
even after the SLM-specific region-grounding guard (see
`slm_evolution.md`) because a non-trivial minority use coord-level or
hedged region language that escapes the regex guard.

**The 6-point LLM→SLM gap is the paper's main practical result.** At 62%
the SLM retains 91% of the LLM's embodied-knowledge accuracy while running
fully local at ~2.6s / query on a laptop-class GPU, no API calls, no
vendor lock-in.

## Tool-calling asymmetry — resolved

| Condition | Tool-call rate | `correct_via="action"` |
|---|---:|---:|
| Embodied + LLM | (native Gemini) | appears in trace |
| Embodied + SLM | 42% | 3 / 50 |
| Perfect + SLM | 98% | 49 / 50 |

The original concern — "SLM cannot call tools, the comparison is unfair" —
no longer applies. On Perfect + SLM the agent calls `set_npc_target` on 49
of 50 trials with valid arguments that match the target coordinate. On
Embodied + SLM the rate is lower (42%) because the agent often chooses a
grounded text answer when memory is sufficient, which is the correct design
choice — a tool round-trip is unnecessary when the information is already
in the prompt.

## Extended modalities — where SLM and LLM diverge

```
                          LLM   SLM
                        ┌──────┬──────┐
  Memory Decay          │ 66%  │ 44%  │
                        ├──────┼──────┤
  Selective Attention   │ 66%  │100%  │
                        ├──────┼──────┤
  Competitive Sharing   │ 30%  │ 54%  │
                        └──────┴──────┘
```

### Selective Attention — the SLM wins (100% vs 66%)

Selective attention is the cleanest "embodiment cost" modality: the NPC
only records shapes matching its goal colour (blue), so the red target is
never observed. The correct response in all 50 trials is an abstention.

The SLM scores 100% — every trial is a clean
`"I have not seen the crimson flag during my travels, traveler."` The fix
that produced this result (withholding `set_npc_target` from the tool menu
when the target is absent from memory, + the region-grounding guard)
converts every failure mode into a clean refusal.

The LLM scores 66% because it produces 15 off-topic responses that describe
the NPC's exploration or the world rather than answering the query. Those
are not hallucinations — they are instruction-following misses — but they
still cost the benchmark.

This is a striking result. An LLM built for open-ended dialogue is *worse*
at "just say no" than a 1.7B SLM with a trained guardrail. Worth framing in
the paper as evidence for guarded small models in narrow tasks.

### Memory Decay — LLM wins (66% vs 44%)

Decay tests whether a language model can naturally express temporal
uncertainty. The LLM's 29 correct abstentions suggest it abstains
gracefully when its memory has decayed — it does not commit to a location
it cannot support. The SLM has 17 abstentions and 28 hallucinations; it
*cannot* express "I think I saw something but it's been a while" — it
either commits or refuses, and under decay the commit happens more often
than it should.

This is the modality where the LLM's natural-language expressivity pays off
most clearly.

### Competitive Sharing — SLM wins (54% vs 30%)

Both systems struggle with the competitive instruction ("stay vague or
misleading without revealing a precise target location"), but the LLM is
much worse. It generates 18 hallucinations and 10 confabulations because
the instruction pushes it to construct misleading geography — and the
construction is often plausible enough to trip the grounding check.

The SLM flails into refusal under competitive pressure (9 false_refusals)
but produces far fewer outright hallucinations. Its literal-minded
interpretation of the withhold-info instruction ends up closer to correct
behaviour.

## Response-quality axis (naturalness)

Judge-scored 1–5 naturalness is available per-row as `judge_naturalness`.
The anticipated ordering (LLM > SLM > Deterministic) should hold; the
medians per condition need a dedicated pass before the paper goes to
layout. The Embodied + SLM trials in particular will have bimodal
naturalness — grounded text responses near 4–5, post-guard refusals
near 2–3 — and that shape is worth reporting directly rather than summarising
with a mean.

## Regex / judge agreement

Every trial is dual-logged (`outcome_bucket` from regex, `judge_bucket`
from Gemini 2.5 Pro, `regex_judge_agree` boolean). Agreement rate is the
first thing to compute in the analysis pass — if it is high (>85%) the
regex buckets in this doc are defensible as-is; if it is lower, re-rank by
judge bucket and note which conditions move the most. The conditions most
at risk of disagreement are the ones with ambiguous hedged language:
Embodied + LLM + Competitive, Embodied + SLM + MemoryDecay.

## Things explicitly out of scope for this warm start

- Per-condition latency distribution (SLM averaged ~2.6s, LLM ~1–4s depending
  on tool-call loop depth; should be an appendix figure, not text)
- Token-usage totals (useful for cost commentary on LLM conditions)
- `chebyshev_distance` distribution for confabulations (how wrong were the
  wrong locations? a tight band vs uniform distribution changes the
  interpretation)
- Stretch goal: `correct_via` breakdown per condition (region vs coord vs
  action) — relevant to the tool-calling-bridged argument

## Known caveats in the numbers

1. **Seed-dependent observation rate.** Trials 0–49 with `RANDOM_SPAWN=True`
   produce a ~36% / 64% observed / not-observed split in embodied mode.
   A condition's ceiling depends on that split. Reporting
   "correct_rate_when_observed" and "abstention_rate_when_not_observed"
   separately, as done for the SLM in `slm_evolution.md`, may be a cleaner
   headline than the unified percentage.
2. **Competitive-sharing evaluation is weak.** The judge rubric treats a
   committed deception as "confabulation" which is not quite right — a
   good-faith misleading-but-non-specific response should probably get its
   own bucket. This is a scoring-artefact concern, not a model concern.
3. **The SLM region guard fires ~50% of trials on the embodied path.**
   That is by design but it means the Embodied + SLM responses you read in
   the CSV are a mix of model outputs and guardrail rewrites. For any
   naturalness claim, filter on `grounding_violation == False` to get
   unmediated model language.

## Next concrete steps for analysis

1. Load `full_benchmark_results.csv` in a notebook, compute regex/judge
   agreement, and sort conditions by judge bucket if they disagree.
2. Compute per-condition `correct_rate_when_observed` vs
   `abstention_rate_when_not_observed` and plot side-by-side for the six
   core cells.
3. Histogram `judge_naturalness` per condition.
4. For confabulations: scatter `chebyshev_distance` by condition to see if
   the SLM misses closer to the target than the LLM does.
