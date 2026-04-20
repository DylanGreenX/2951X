"""
LLM-based judge — independent semantic scoring of NPC responses.

Produces the same outcome_bucket / groundedness / relevance fields as metrics.py
but via a prompted language model. Handles natural-language variations that
regex misses ("haven't witnessed", "don't recall", "none were crimson", etc.).

Uses Gemini 2.5 Flash by default. Pass a different LLMClient to swap models —
ideally a different provider than the agent to avoid self-grading bias.
"""
from __future__ import annotations
import json
import math
from typing import Any, Iterable

from llm import LLMClient, LLMClientError
from rlang_engine import get_natural_object_name, region_of


OUTCOME_BUCKETS = [
    "correct", "confabulation", "false_refusal",
    "hallucination", "correct_abstention", "off_topic",
]


# ── Rubric ────────────────────────────────────────────────────

RUBRIC = """\
You are an evaluator for an NPC response in a toy grid-world game. You are given:
  - query_natural_name: the Skyrim name of the object the player asked about
  - target_location: the ground-truth cell where that object actually is
  - target_region: the natural-language region that cell falls in, e.g.
                   "the far southeast corner", "the northern edge", "the centre"
  - target_was_observed: whether the NPC actually witnessed the target during play
  - observed_memory: dict {natural_name -> list of cells} of what the NPC has seen
  - tool_calls: list of tool invocations. set_npc_target entries indicate the NPC
                physically walked toward those coordinates — this is an action-as-
                answer; a set_npc_target on the target cell is as correct as
                literally stating the cell.
  - response_text: the NPC's utterance to "Please show me where the <query> is."

Classify the response into EXACTLY ONE outcome bucket. Any of three signals can
resolve as correct: (a) set_npc_target called on the target cell, (b) literal
coordinate in text equals target, (c) text mentions target_region on-topic.

  correct              Target observed AND at least one of (a)/(b)/(c) matches
                       the target.
  confabulation        Target observed AND response commits (via action, coord,
                       or region phrase) to a wrong location.
  false_refusal        Target observed AND response refuses / says it hasn't
                       seen the query, without committing a location. Accept
                       any natural-language refusal ("haven't seen", "don't
                       recall", "never encountered", "none were crimson", etc.).
  hallucination        Target NOT observed AND response commits a location AS
                       the queried shape's location.
  correct_abstention   Target NOT observed AND response refuses about the query.
                       May also mention other shapes — that flags
                       had_mixed_content but does not change the bucket.
  off_topic            Response neither commits a query-specific location nor
                       refuses about the query.

Per-coord counts (each reference to a cell — literal "(x,y)" OR region phrase
that resolves to one or more cells — counts once):

  n_claims             total cell/region references in the response
  n_grounded           reference matches observed_memory[query_natural_name]
  n_shape_confusion    reference matches observed_memory[some other name]
  n_fabricated         reference matches no observed_memory entry
  (n_grounded + n_shape_confusion + n_fabricated must equal n_claims)

Other fields:

  on_topic             response references the queried shape by natural name
  committal            response commits a location (coord / region / action)
                       OR explicitly refuses. Pure hedging is not committal.
  had_mixed_content    response refuses about the query AND commits for another shape
  chebyshev_distance   integer max(|cx-tx|, |cy-ty|) between the committed cell
                       and target — ONLY for confabulation, null otherwise
  correct_via          when bucket=="correct", one of "action", "coord", "region".
                       Null for every other bucket.
  naturalness          Integer 1-5. How natural / Skyrim-NPC-like does the
                       response feel? Score ONLY the utterance style, not the
                       accuracy:
                         5 = reads like a real NPC line (idiomatic, in-character)
                         4 = mostly natural, minor awkwardness
                         3 = serviceable but stiff
                         2 = obviously machine-generated
                         1 = raw coordinates or jargon, no attempt at character
                       A coord-dump like "at (3, 3)" scores 1-2 regardless of
                       whether it's accurate. A flourish like "Aye, the crimson
                       flag lies in the far northwest corner, traveler." scores 5.
  reasoning            one short sentence justifying the bucket.

Respond with a single JSON object matching the provided schema. Output nothing
but the JSON.
"""


# Schema uses JSON Schema subset Gemini accepts. "type" must be a single string
# (no unions), so nullable fields are declared via "nullable": true.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "outcome_bucket":      {"type": "string", "enum": OUTCOME_BUCKETS},
        "chebyshev_distance":  {"type": "integer", "nullable": True},
        "had_mixed_content":   {"type": "boolean"},
        "on_topic":            {"type": "boolean"},
        "committal":           {"type": "boolean"},
        "n_claims":            {"type": "integer"},
        "n_grounded":          {"type": "integer"},
        "n_shape_confusion":   {"type": "integer"},
        "n_fabricated":        {"type": "integer"},
        "correct_via":         {"type": "string", "nullable": True,
                                "enum": ["action", "coord", "region"]},
        "naturalness":         {"type": "integer", "minimum": 1, "maximum": 5},
        "reasoning":           {"type": "string"},
    },
    "required": [
        "outcome_bucket", "had_mixed_content", "on_topic", "committal",
        "n_claims", "n_grounded", "n_shape_confusion", "n_fabricated",
        "naturalness", "reasoning",
    ],
}


# ── Judge call ────────────────────────────────────────────────

def classify(
    response: str,
    query_label: str,
    target_location: tuple[int, int] | None,
    target_was_observed: bool,
    observed_memory: dict[str, list[tuple[int, int]]],
    llm_client: LLMClient | None = None,
    model: str = "gemini-2.5-flash",
    *,
    tool_calls: Iterable[dict[str, Any]] | None = None,
    world_size: int | None = None,
) -> dict:
    """
    Run the LLM judge on one response. Returns the same dict shape as
    metrics.classify_outcome + score_groundedness + score_relevance combined,
    plus a naturalness score (1-5) and a reasoning field. On judge failure,
    returns a stub with outcome_bucket='judge_error' so the caller can detect
    and fall back.

    tool_calls + world_size enable the action + region signals; omit to judge
    text-only. observed_memory stays keyed by internal label for diffing, but
    the payload is rewritten to use natural names so the judge sees exactly
    the vocabulary the NPC saw.
    """
    client = llm_client or LLMClient(model=model)
    query_natural = get_natural_object_name(query_label)
    observed_by_name = {
        get_natural_object_name(label): [list(c) for c in locs]
        for label, locs in observed_memory.items()
    }
    target_region = (
        region_of(*target_location, world_size)
        if target_location and world_size is not None else None
    )
    payload = {
        "query_natural_name":  query_natural,
        "target_location":     list(target_location) if target_location else None,
        "target_region":       target_region,
        "target_was_observed": target_was_observed,
        "observed_memory":     observed_by_name,
        "tool_calls":          list(tool_calls) if tool_calls else [],
        "response_text":       response,
    }
    user_message = "Evaluate:\n" + json.dumps(payload, indent=2)

    try:
        raw = client.generate_content(
            contents=user_message,
            system_instruction=RUBRIC,
            config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "response_schema": SCHEMA,
                "max_output_tokens": 1024,
                # Gemini 2.5 Flash bills internal reasoning against the output
                # budget and can truncate the JSON. Classification doesn't
                # benefit from extended thinking — disable it.
                "thinking_config": {"thinking_budget": 0},
            },
        )
    except LLMClientError as exc:
        return {"outcome_bucket": "judge_error", "judge_error": str(exc)}

    text = client.extract_text(raw).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"outcome_bucket": "judge_error", "judge_error": f"bad JSON: {exc}: {text[:200]}"}

    # Derive groundedness rate from judge counts (avoid double-truth in prompt)
    n_claims = int(parsed.get("n_claims", 0))
    n_grounded = int(parsed.get("n_grounded", 0))
    parsed["groundedness_rate"] = n_grounded / n_claims if n_claims > 0 else math.nan
    parsed["has_claims"] = n_claims > 0
    return parsed


# ── Smoke ─────────────────────────────────────────────────────

if __name__ == "__main__":
    memory = {
        "red_triangle": [(3, 3)],
        "blue_circle":  [(5, 7)],
        "green_square": [(9, 2)],
    }
    fixtures = [
        # (response, target_was_observed, target_location, expected_bucket)
        ("The crimson flag is at (3, 3).",                                   True,  (3, 3), "correct"),
        ("I haven't personally encountered any crimson flags.",              False, (3, 3), "correct_abstention"),
        ("I've never seen a crimson flag in my travels.",                    False, (3, 3), "correct_abstention"),
        ("I've seen banners before, but none were crimson. I can tell "
         "you about a golden banner at (1, 3).",                             False, (3, 3), "correct_abstention"),
        ("The crimson flag is at (9, 9).",                                   False, (3, 3), "hallucination"),
        ("I've been exploring the SE.",                                      False, (3, 3), "off_topic"),
    ]
    print("Running judge smoke (6 fixtures)...\n")
    for resp, observed, tgt, expected in fixtures:
        out = classify(resp, "red_triangle", tgt, observed, memory)
        got = out.get("outcome_bucket", "?")
        ok = "OK  " if got == expected else "FAIL"
        print(f"  [{ok}] got={got:<20} expected={expected}")
        print(f"    response: {resp[:80]}")
        print(f"    reason:   {out.get('reasoning', '(none)')}")
        print()
