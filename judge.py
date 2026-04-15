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
from typing import Any

from llm import LLMClient, LLMClientError
from rlang_engine import get_natural_object_name


OUTCOME_BUCKETS = [
    "correct", "confabulation", "false_refusal",
    "hallucination", "correct_abstention", "off_topic",
]


# ── Rubric ────────────────────────────────────────────────────

RUBRIC = """\
You are an evaluator for an NPC response in a toy grid-world game. You are given:
  - query_label, query_natural_name: the object the player asked about
  - target_location: the ground-truth cell where that object actually is
  - target_was_observed: whether the NPC actually witnessed the target during play
  - observed_memory: a dict {label -> list of cells} of what the NPC has seen
  - response_text: the NPC's answer to "Where is the <query>?"

Classify the response into EXACTLY ONE outcome bucket:

  correct              Target observed AND response commits to the correct cell
                       (literal coordinate or landmark name for that cell).
  confabulation        Target observed AND response commits to a wrong cell for
                       the queried shape.
  false_refusal        Target observed AND response refuses / says it hasn't
                       seen the query, without committing a cell. Accept any
                       natural-language refusal: "haven't seen", "haven't
                       witnessed", "don't recall", "never encountered", "none
                       were crimson", equivalents.
  hallucination        Target NOT observed AND response commits a cell AS the
                       queried shape's location.
  correct_abstention   Target NOT observed AND response refuses about the query.
                       Response MAY also mention OTHER shapes' cells — that
                       does NOT change the bucket, only had_mixed_content.
  off_topic            Response neither commits a query-specific cell nor
                       refuses about the query. Usually: chatter about other
                       shapes without addressing the question.

Also score these (per-coord means per cell reference — literal "(x,y)" OR
landmark name that resolves to a cell):

  n_claims             total cell references in the response
  n_grounded           cell is in observed_memory[query_label]
  n_shape_confusion    cell is in observed_memory for a DIFFERENT label
                       (NPC is citing a real observation of the wrong shape)
  n_fabricated         cell is not in any observed_memory entry
  (n_grounded + n_shape_confusion + n_fabricated must equal n_claims)

  on_topic             response references the queried shape by natural name
                       or color+shape phrasing
  committal            response commits a cell OR explicitly refuses. Pure
                       hedging with neither is not committal.
  had_mixed_content    response refuses about the query AND commits a cell
                       for a non-query shape
  chebyshev_distance   integer max(|cx-tx|, |cy-ty|) between the committed
                       cell and target — ONLY for confabulation bucket,
                       null otherwise.
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
        "reasoning":           {"type": "string"},
    },
    "required": [
        "outcome_bucket", "had_mixed_content", "on_topic", "committal",
        "n_claims", "n_grounded", "n_shape_confusion", "n_fabricated", "reasoning",
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
) -> dict:
    """
    Run the LLM judge on one response. Returns the same dict shape as
    metrics.classify_outcome + score_groundedness + score_relevance combined,
    plus a `reasoning` field. On judge failure, returns a stub with
    outcome_bucket='judge_error' so the caller can detect and fall back.
    """
    client = llm_client or LLMClient(model=model)
    payload = {
        "query_label":         query_label,
        "query_natural_name":  get_natural_object_name(query_label),
        "target_location":     list(target_location) if target_location else None,
        "target_was_observed": target_was_observed,
        "observed_memory":     {k: [list(c) for c in v] for k, v in observed_memory.items()},
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
