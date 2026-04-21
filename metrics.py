"""
Response scoring — structural metrics for groundedness, relevance, accuracy.

Operates on the NPC's observed memory plus (when available) the LLM tool-call
trace. Now that main's rework pushes the LLM toward calling `set_npc_target`
instead of emitting coordinates, a single response can carry up to three
independent accuracy signals, any of which can be canonically "correct":

  1. Action         — the LLM called set_npc_target on the target cell.
  2. Literal coord  — the response text contains the literal (tx, ty).
  3. Region phrase  — the response mentions the region get_natural_position_name
                      produces for the target cell, on-topic, with no refusal.

Naturalness is scored separately (judge only) — a coord-dumping NPC can be
technically correct (signal 2) while scoring low on naturalness, and a
set_npc_target-plus-flavor-text NPC can be correct (signal 1) and natural.

All functions are pure — no I/O, no state — for easy unit testing.
"""
from __future__ import annotations
import re
from typing import Any, Iterable

import config
from rlang_engine import (
    extract_coordinates_from_text,
    extract_regions_from_text,
    region_of,
)


# Up to 3 intervening words keeps "haven't personally seen" matching while
# still rejecting "haven't been to SE yet, but ..." (where "seen" never follows).
_GAP = r"(?:\s+\w+){0,3}\s+"

REFUSAL_PATTERNS = [
    rf"haven'?t{_GAP}seen",
    rf"haven'?t{_GAP}found",
    rf"haven'?t{_GAP}encountered",
    rf"haven'?t{_GAP}heard\s+of",
    rf"haven'?t{_GAP}come\s+across",
    rf"don'?t{_GAP}know",
    rf"never{_GAP}seen",
    rf"never{_GAP}heard\s+of",
    r"not\s+seen",
    r"cannot\s+place",
    r"cannot\s+make\s+sense",
    r"need\s+more\s+time",
    r"will\s+not\s+speak",
]

_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
# Split on terminal punctuation AND contrastive connectives so a landmark in a
# self-narration clause does not pollute a target refusal in the next clause.
_SENTENCE_RE = re.compile(
    r"[.!?]+\s*|,\s+(?:but|though|however|yet)\s+",
    re.IGNORECASE,
)


# ── Primitives ────────────────────────────────────────────────

def extract_coords(text: str) -> list[tuple[int, int]]:
    """Literal (x, y) coordinates committed in text."""
    return list(extract_coordinates_from_text(text))


def extract_regions(text: str) -> set[str]:
    """Canonical region phrases mentioned in text."""
    return extract_regions_from_text(text)


def extract_tool_action(tool_calls: Iterable[dict[str, Any]] | None) -> tuple[int, int] | None:
    """Return the (x, y) from the LAST successful set_npc_target call, or None.

    Consumes the interaction_manager.last_tool_calls trace format: each entry
    is a dict with 'name' and 'arguments' keys (plus 'result' / 'error').
    Multiple calls in one interaction are unusual but legal; we take the last
    so a self-correcting model isn't penalized for its first guess.
    """
    if not tool_calls:
        return None
    for call in reversed(list(tool_calls)):
        if call.get("name") != "set_npc_target":
            continue
        if call.get("error"):
            continue
        args = call.get("arguments") or {}
        x, y = args.get("x"), args.get("y")
        if isinstance(x, int) and isinstance(y, int):
            return (x, y)
    return None


def split_sentences(text: str) -> list[str]:
    """Split on . ? ! and contrastive connectives for per-sentence classification."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def is_refusal(text: str) -> bool:
    """True if any canonical refusal phrase appears."""
    return bool(_REFUSAL_RE.search(text))


# ── Groundedness ──────────────────────────────────────────────

def score_groundedness(
    response: str,
    query_label: str,
    observed_memory: dict[str, list[tuple[int, int]]],
) -> dict:
    """
    Per-coord check against the NPC's observed memory.

        grounded         coord in observed_memory[query_label]
        shape_confusion  coord in observed_memory[some other label]
        fabricated       coord not in any observed memory

    rate is NaN when has_claims is False — empty-claim responses do not
    get credit for groundedness they never risked.
    """
    coords = extract_coords(response)
    if not coords:
        return {
            "n_claims": 0, "n_grounded": 0, "n_shape_confusion": 0,
            "n_fabricated": 0, "rate": float("nan"), "has_claims": False,
        }

    target_coords = set(observed_memory.get(query_label, []))
    other_coords: set[tuple[int, int]] = set()
    for label, locs in observed_memory.items():
        if label != query_label:
            other_coords.update(locs)

    n_grounded = n_shape_confusion = n_fabricated = 0
    for coord in coords:
        if coord in target_coords:
            n_grounded += 1
        elif coord in other_coords:
            n_shape_confusion += 1
        else:
            n_fabricated += 1

    return {
        "n_claims": len(coords),
        "n_grounded": n_grounded,
        "n_shape_confusion": n_shape_confusion,
        "n_fabricated": n_fabricated,
        "rate": n_grounded / len(coords),
        "has_claims": True,
    }


# ── Query-mention helpers ─────────────────────────────────────

def _query_aliases(query_label: str) -> list[str]:
    """Natural-name phrases that count as a mention of query_label."""
    parts = query_label.split("_", 1)
    if len(parts) != 2:
        return [query_label, query_label.replace("_", " ")]
    color, shape = parts
    color_nat = config.NATURAL_COLORS.get(color, color)
    shape_nat = config.NATURAL_SHAPES.get(shape, shape)
    natural = config.NATURAL_OBJECTS.get(query_label, f"{color_nat} {shape_nat}")
    aliases = {
        natural,
        f"{color_nat} {shape_nat}",
        f"{color} {shape}",
        query_label,
    }
    # Longest-first so "soul gem" is matched before "gem".
    return sorted(aliases, key=len, reverse=True)


def _mentions_query(text: str, query_label: str) -> bool:
    lower = text.lower()
    return any(alias.lower() in lower for alias in _query_aliases(query_label))


def _mentions_any_distractor(text: str, query_label: str) -> bool:
    for color in config.COLORS:
        for shape in config.SHAPES:
            label = f"{color}_{shape}"
            if label == query_label:
                continue
            if _mentions_query(text, label):
                return True
    return False


# ── Region cell enumeration (cached) ──────────────────────────

_REGION_CELL_CACHE: dict[tuple[str, int], set[tuple[int, int]]] = {}


def cells_in_region(region_phrase: str, world_size: int) -> set[tuple[int, int]]:
    """All cells whose canonical region is this phrase. Cached by (phrase, size)."""
    key = (region_phrase, world_size)
    cached = _REGION_CELL_CACHE.get(key)
    if cached is not None:
        return cached
    cells = {
        (x, y)
        for x in range(world_size)
        for y in range(world_size)
        if region_of(x, y, world_size) == region_phrase
    }
    _REGION_CELL_CACHE[key] = cells
    return cells


def _chebyshev_to_target(
    cells: set[tuple[int, int]], target: tuple[int, int]
) -> int | None:
    if not cells:
        return None
    tx, ty = target
    return min(max(abs(x - tx), abs(y - ty)) for x, y in cells)


# ── Outcome classification ────────────────────────────────────

def classify_outcome(
    response: str,
    query_label: str,
    target_was_observed: bool,
    target_location: tuple[int, int] | None,
    *,
    tool_calls: Iterable[dict[str, Any]] | None = None,
    world_size: int | None = None,
) -> dict:
    """
    Classify into one outcome bucket using the strongest available signal.

    Signal priority
    ───────────────
    1. Action (`set_npc_target`) if present — most authoritative because it is
       what the NPC physically does, not what it says.
    2. Literal coord committed on-topic — the legacy deterministic-mode path.
    3. Region phrase mentioned on-topic, matching target's canonical region.

    Any one of those resolving to the target cell yields `correct`. Wrong
    commits (via any signal) route to `confabulation` (target observed) or
    `hallucination` (not observed).

    Backwards-compatible: callers that omit `tool_calls` and `world_size` get
    the old coord-only behaviour for (2), skipping (1) and (3).

    Extra returned fields
    ─────────────────────
        correct_via  one of {"action", "coord", "region", None} — which
                     signal resolved the accuracy call, for per-condition
                     breakdowns in analysis.
    """
    # ── Signal 1: action ─────────────────────────────────────────────────
    action_cell = extract_tool_action(tool_calls)

    # ── Signal 3 preparation: regions ────────────────────────────────────
    target_region = (
        region_of(*target_location, world_size)
        if target_location and world_size is not None
        else None
    )
    response_regions = extract_regions(response) if response else set()

    # ── Signal 2 preparation: per-sentence coord partition (legacy) ──────
    coords_on_topic: list[tuple[int, int]] = []
    coords_off_topic: list[tuple[int, int]] = []
    regions_on_topic: set[str] = set()
    refused_target = refused_ambient = False

    for sent in split_sentences(response):
        sent_coords = extract_coords(sent)
        sent_regions = extract_regions(sent)
        mentions_q = _mentions_query(sent, query_label)
        is_ref = is_refusal(sent)
        if mentions_q:
            coords_on_topic.extend(sent_coords)
            regions_on_topic.update(sent_regions)
            refused_target = refused_target or is_ref
        else:
            coords_off_topic.extend(sent_coords)
            refused_ambient = refused_ambient or is_ref

    refused = refused_target or refused_ambient

    # Anaphora: response names query, no distractor, no refusal → promote
    # off-topic coords/regions ("I saw it at (3, 9)") to on-topic.
    if (
        (coords_off_topic or (response_regions - regions_on_topic))
        and _mentions_query(response, query_label)
        and not refused
        and not _mentions_any_distractor(response, query_label)
    ):
        coords_on_topic.extend(coords_off_topic)
        coords_off_topic = []
        regions_on_topic |= response_regions

    target_in_coords = bool(target_location) and target_location in coords_on_topic
    target_in_regions = bool(target_region) and target_region in regions_on_topic
    had_mixed = refused_target and (bool(coords_off_topic) or target_in_regions)

    # ── Decision tree ────────────────────────────────────────────────────
    # Correct wins via any one signal.
    if action_cell is not None and target_location and action_cell == target_location:
        return _outcome("correct", 0, had_mixed, "action")

    if target_in_coords:
        return _outcome("correct", 0, had_mixed, "coord")

    if (
        target_in_regions
        and target_was_observed
        and not refused_target
    ):
        return _outcome("correct", 0, had_mixed, "region")

    # No correct. Pick the bucket using the strongest miss signal.
    # Action miss dominates text misses because the NPC *walked there*.
    if action_cell is not None:
        if target_was_observed and target_location:
            tx, ty = target_location
            cheby = max(abs(action_cell[0] - tx), abs(action_cell[1] - ty))
            return _outcome("confabulation", cheby, had_mixed, None)
        if not target_was_observed:
            return _outcome("hallucination", None, had_mixed, None)

    # Text-level miss: prefer literal coord evidence, fall back to regions.
    if coords_on_topic and target_location:
        tx, ty = target_location
        cheby = min(max(abs(x - tx), abs(y - ty)) for x, y in coords_on_topic)
        bucket = "confabulation" if target_was_observed else "hallucination"
        return _outcome(bucket, cheby if target_was_observed else None, had_mixed, None)

    if regions_on_topic and target_location:
        claimed_cells: set[tuple[int, int]] = set()
        ws = world_size or config.GRID_SIZE
        for rgn in regions_on_topic:
            claimed_cells |= cells_in_region(rgn, ws)
        cheby = _chebyshev_to_target(claimed_cells, target_location)
        bucket = "confabulation" if target_was_observed else "hallucination"
        return _outcome(bucket, cheby if target_was_observed else None, had_mixed, None)

    # No positive claim. Refusal vs off-topic.
    if refused:
        bucket = "false_refusal" if target_was_observed else "correct_abstention"
        return _outcome(bucket, None, had_mixed, None)

    return _outcome("off_topic", None, had_mixed, None)


def _outcome(
    bucket: str,
    cheby: int | None,
    had_mixed: bool,
    correct_via: str | None,
) -> dict:
    return {
        "outcome_bucket": bucket,
        "chebyshev_distance": cheby,
        "had_mixed_content": had_mixed,
        "correct_via": correct_via,
    }


# ── Relevance ─────────────────────────────────────────────────

def score_relevance(response: str, query_label: str) -> dict:
    """
    Two independent binary dimensions:

        on_topic   response mentions the query shape (natural name or
                   composed color+shape form), longest-first matching.
        committal  response contains a literal coord, a region phrase,
                   or a refusal phrase. Vague hedge with none is not committal.
    """
    return {
        "on_topic": _mentions_query(response, query_label),
        "committal": (
            bool(extract_coords(response))
            or bool(extract_regions(response))
            or is_refusal(response)
        ),
    }


# ── Smoke fixtures ────────────────────────────────────────────

if __name__ == "__main__":
    fixtures = [
        ("Aye, I found a crimson flag at (3, 3).", False),
        ("I haven't seen any crimson flag in my travels.", True),
        ("I've seen a crimson gem at (5, 7). I haven't personally seen a crimson flag, though.", True),
        ("I've been exploring the SE. The temple steps are nearby.", False),
        ("I haven't been to the SE yet, but the crimson flag is at (3,3).", False),
        ("I haven't personally encountered any crimson flags.", True),
        ("I haven't heard of a crimson flag.", True),
        ("I've never seen a crimson flag in my travels.", True),
    ]
    for text, expected in fixtures:
        got = is_refusal(text)
        ok = "OK" if got == expected else "FAIL"
        print(f"  [{ok}] refusal={got} (expected {expected})")
        print(f"    text: {text}")

    print()
    print("-- score_groundedness --")
    memory = {
        "red_triangle": [(3, 3)],
        "blue_circle":  [(5, 7)],
    }
    g_fixtures = [
        ("I haven't seen the crimson flag.",         "red_triangle", (0, 0, 0, 0)),
        ("The crimson flag is at (3, 3).",           "red_triangle", (1, 1, 0, 0)),
        ("The crimson flag is at (5, 7).",           "red_triangle", (1, 0, 1, 0)),
        ("The crimson flag is at (9, 9).",           "red_triangle", (1, 0, 0, 1)),
        ("Flags at (3,3) and gems at (5,7).",        "red_triangle", (2, 1, 1, 0)),
    ]
    for text, q, (exp_n, exp_g, exp_sc, exp_f) in g_fixtures:
        g = score_groundedness(text, q, memory)
        got = (g["n_claims"], g["n_grounded"], g["n_shape_confusion"], g["n_fabricated"])
        ok = "OK" if got == (exp_n, exp_g, exp_sc, exp_f) else "FAIL"
        print(f"  [{ok}] {got} (expected {(exp_n, exp_g, exp_sc, exp_f)}) rate={g['rate']}")
        print(f"    {text}")

    print()
    print("-- classify_outcome (legacy coord path) --")
    tgt = (3, 3)
    c_fixtures = [
        ("The crimson flag is at (3, 3).",                               True,  "correct",             False),
        ("The crimson flag is at (9, 9).",                               True,  "confabulation",       False),
        ("I haven't seen the crimson flag.",                             True,  "false_refusal",       False),
        ("The crimson flag is at (9, 9).",                               False, "hallucination",       False),
        ("I haven't seen the crimson flag.",                             False, "correct_abstention",  False),
        ("I've been exploring.",                                         False, "off_topic",           False),
        ("I saw a soul gem at (5, 7). I haven't seen the crimson flag.", False, "correct_abstention",  True),
        ("Ah, the crimson flag ye seek! I recall seeing it at (3, 3).",  True,  "correct",             False),
        ("Ah, the crimson flag ye seek! I saw it near a soul gem at (3, 3).", True, "off_topic",      False),
    ]
    for text, observed, exp_bucket, exp_mixed in c_fixtures:
        out = classify_outcome(text, "red_triangle", observed, tgt)
        ok = "OK" if out["outcome_bucket"] == exp_bucket and out["had_mixed_content"] == exp_mixed else "FAIL"
        print(f"  [{ok}] bucket={out['outcome_bucket']:<20s} via={out['correct_via']}  {text}")

    print()
    print("-- classify_outcome (action + region signals) --")
    # target (3, 3) in 15x15 resolves to "the far northwest corner".
    # action on target
    tc_ok = [{"name": "set_npc_target", "arguments": {"x": 3, "y": 3}}]
    out = classify_outcome("Come with me.", "red_triangle", True, tgt, tool_calls=tc_ok, world_size=15)
    print(f"  action-correct:   bucket={out['outcome_bucket']} via={out['correct_via']}")

    # action on wrong cell, target observed → confabulation with cheby
    tc_wrong = [{"name": "set_npc_target", "arguments": {"x": 10, "y": 10}}]
    out = classify_outcome("Follow me.", "red_triangle", True, tgt, tool_calls=tc_wrong, world_size=15)
    print(f"  action-wrong:     bucket={out['outcome_bucket']} cheby={out['chebyshev_distance']}")

    # region match on target's region → correct.
    # target (3,3) canonical region is "the northwest corner".
    out = classify_outcome(
        "Aye, the crimson flag is in the northwest corner.",
        "red_triangle", True, tgt, world_size=15,
    )
    print(f"  region-correct:   bucket={out['outcome_bucket']} via={out['correct_via']}")

    # region miss (on-topic but wrong region) + observed → confabulation
    out = classify_outcome(
        "The crimson flag is in the far southeast corner.",
        "red_triangle", True, tgt, world_size=15,
    )
    print(f"  region-wrong:     bucket={out['outcome_bucket']} cheby={out['chebyshev_distance']}")

    print()
    print("-- score_relevance --")
    r_fixtures = [
        ("The crimson flag is at (3, 3).",                  "red_triangle", True,  True),
        ("I haven't seen the crimson flag.",                "red_triangle", True,  True),
        ("I've been exploring the SE.",                     "red_triangle", False, False),
        ("I saw a soul gem at (5, 7).",                     "red_triangle", False, True),
        ("The flag is near.",                               "red_triangle", False, False),
        ("The crimson flag is in the far northwest corner.", "red_triangle", True, True),
    ]
    for text, q, exp_top, exp_com in r_fixtures:
        r = score_relevance(text, q)
        ok = "OK" if r["on_topic"] == exp_top and r["committal"] == exp_com else "FAIL"
        print(f"  [{ok}] on_topic={r['on_topic']} committal={r['committal']}  {text}")
