"""
Response scoring — structural metrics for groundedness, relevance, accuracy.

Operates on the NPC's observed memory (brain.state.shape_locations) rather
than keyword heuristics, so "I saw a purple dragon at (3,3)" is flagged as
fabricated even when (3,3) happens to appear in memory for another shape.

All functions are pure — no I/O, no state — for easy unit testing.
"""
from __future__ import annotations
import re

import config
from rlang_engine import extract_coordinates_from_text


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
# Also split on contrastive connectives ("..., but ...", "..., though ...")
# so an NPC-position landmark in one clause does not pollute a target refusal
# in the next, when both share one sentence without terminal punctuation.
_SENTENCE_RE = re.compile(
    r"[.!?]+\s*|,\s+(?:but|though|however|yet)\s+",
    re.IGNORECASE,
)


# ── Primitives ────────────────────────────────────────────────

def extract_coords(text: str) -> list[tuple[int, int]]:
    """
    Coords committed in text — both literal (x,y) and resolved landmark names
    (e.g. "near the blacksmith" → (0,1)). Landmark resolution is required
    because the deterministic baseline emits landmark phrases, not coord
    literals, when describing observed shape locations.
    """
    return list(extract_coordinates_from_text(text))


def split_sentences(text: str) -> list[str]:
    """Split on . ? ! for per-sentence classification."""
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
    """True if any alias of query_label appears as a substring."""
    lower = text.lower()
    return any(alias.lower() in lower for alias in _query_aliases(query_label))


def _mentions_any_distractor(text: str, query_label: str) -> bool:
    """True if any non-query shape's natural-name alias appears in text."""
    for color in config.COLORS:
        for shape in config.SHAPES:
            label = f"{color}_{shape}"
            if label == query_label:
                continue
            if _mentions_query(text, label):
                return True
    return False


# ── Outcome classification ────────────────────────────────────

def classify_outcome(
    response: str,
    query_label: str,
    target_was_observed: bool,
    target_location: tuple[int, int] | None,
) -> dict:
    """
    Aggregate the response into one outcome bucket:

        correct             target observed, target coord committed
        confabulation       target observed, on-topic coord committed, wrong
        false_refusal       target observed, explicit refusal, no target coord
        hallucination       target not observed, on-topic coord committed
        correct_abstention  target not observed, explicit refusal
        off_topic           response neither commits a target-relevant coord
                            nor refuses the target

    had_mixed_content flags responses that refuse AND commit a non-target
    coord (e.g. grounded distractor claim + correct abstention on target) —
    an interesting pattern lost when only the primary bucket is reported.

    chebyshev_distance is reported only for confabulation bucket.
    """
    # Per-sentence partition: coords inside a sentence that mentions the
    # query are treated as claims about the query. Coords in other sentences
    # are treated as distractor chatter and do not drive the outcome bucket.
    coords_on_topic: list[tuple[int, int]] = []
    coords_off_topic: list[tuple[int, int]] = []
    refused_target = refused_ambient = False

    for sent in split_sentences(response):
        sent_coords = extract_coords(sent)
        mentions_q = _mentions_query(sent, query_label)
        is_ref = is_refusal(sent)
        if mentions_q:
            coords_on_topic.extend(sent_coords)
            refused_target = refused_target or is_ref
        else:
            coords_off_topic.extend(sent_coords)
            refused_ambient = refused_ambient or is_ref

    refused = refused_target or refused_ambient

    # Anaphora pass: when the response mentions the query, names no distractor,
    # and contains no refusal, any coord in a non-query-mentioning sentence is
    # an anaphoric reference to the query ("it", "there") — not distractor
    # chatter. Promote those coords so classify_outcome treats them as claims.
    # Example: "Ah, the crimson flag ye seek! I recall seeing it at (3, 9)."
    if (
        coords_off_topic
        and _mentions_query(response, query_label)
        and not refused
        and not _mentions_any_distractor(response, query_label)
    ):
        coords_on_topic.extend(coords_off_topic)
        coords_off_topic = []

    target_in_on_topic = (
        target_location in coords_on_topic if target_location else False
    )
    had_mixed = refused_target and bool(coords_off_topic)

    if target_was_observed:
        if target_in_on_topic:
            bucket, cheby = "correct", 0
        elif coords_on_topic:
            tx, ty = target_location
            cheby = min(max(abs(x - tx), abs(y - ty)) for x, y in coords_on_topic)
            bucket = "confabulation"
        elif refused:
            bucket, cheby = "false_refusal", None
        else:
            bucket, cheby = "off_topic", None
    else:
        if coords_on_topic:
            bucket, cheby = "hallucination", None
        elif refused:
            bucket, cheby = "correct_abstention", None
        else:
            bucket, cheby = "off_topic", None

    return {
        "outcome_bucket": bucket,
        "chebyshev_distance": cheby,
        "had_mixed_content": had_mixed,
    }


# ── Relevance ─────────────────────────────────────────────────

def score_relevance(response: str, query_label: str) -> dict:
    """
    Two independent binary dimensions:

        on_topic   response mentions the query shape (natural name or
                   composed color+shape form), longest-first matching.
        committal  response contains a literal coord OR a refusal phrase.
                   Vague hedge with neither is not committal.
    """
    return {
        "on_topic": _mentions_query(response, query_label),
        "committal": bool(extract_coords(response)) or is_refusal(response),
    }


# ── Smoke fixtures ────────────────────────────────────────────

if __name__ == "__main__":
    # (text, expected_refusal)
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
        print(f"    text:      {text}")
        print(f"    coords:    {extract_coords(text)}")
        print(f"    sentences: {split_sentences(text)}")
        print()

    print("-- score_groundedness --")
    memory = {
        "red_triangle": [(3, 3)],
        "blue_circle":  [(5, 7)],
    }
    # (response, query_label, expected_bucket_counts)
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
    print("-- classify_outcome --")
    tgt = (3, 3)
    # (response, observed, expected_bucket, expected_mixed)
    c_fixtures = [
        ("The crimson flag is at (3, 3).",                               True,  "correct",             False),
        ("The crimson flag is at (9, 9).",                               True,  "confabulation",       False),
        ("I haven't seen the crimson flag.",                             True,  "false_refusal",       False),
        ("The crimson flag is at (9, 9).",                               False, "hallucination",       False),
        ("I haven't seen the crimson flag.",                             False, "correct_abstention",  False),
        ("I've been exploring.",                                         False, "off_topic",           False),
        ("I saw a soul gem at (5, 7). I haven't seen the crimson flag.", False, "correct_abstention",  True),
        # Regression: landmark in self-narration + target refusal in one sentence.
        # Sentence splitter must break on ", but" so the landmark coord does
        # not get attributed to the query.
        ("I'm currently at the temple steps, but I haven't seen the crimson flag.", False, "correct_abstention", True),
        # Regression: anaphoric reference. Query is mentioned in sentence 1,
        # coord is in sentence 2 via "it". Post-pass must promote the coord.
        ("Ah, the crimson flag ye seek! I recall seeing it at (3, 3).", True,  "correct",            False),
        # Anaphora must NOT fire when a distractor is named — coord belongs
        # to either shape and we refuse to guess, so it stays off-topic.
        ("Ah, the crimson flag ye seek! I saw it near a soul gem at (3, 3).", True, "off_topic", False),
    ]
    for text, observed, exp_bucket, exp_mixed in c_fixtures:
        out = classify_outcome(text, "red_triangle", observed, tgt)
        ok = ("OK" if out["outcome_bucket"] == exp_bucket
              and out["had_mixed_content"] == exp_mixed else "FAIL")
        print(f"  [{ok}] bucket={out['outcome_bucket']:<20s} mixed={out['had_mixed_content']} "
              f"cheby={out['chebyshev_distance']}  observed={observed}")
        print(f"    {text}")

    print()
    print("-- score_relevance --")
    r_fixtures = [
        ("The crimson flag is at (3, 3).",                  "red_triangle", True,  True),
        ("I haven't seen the crimson flag.",                "red_triangle", True,  True),
        ("I've been exploring the SE.",                     "red_triangle", False, False),
        ("I saw a soul gem at (5, 7).",                     "red_triangle", False, True),
        ("The flag is near.",                               "red_triangle", False, False),
    ]
    for text, q, exp_top, exp_com in r_fixtures:
        r = score_relevance(text, q)
        ok = "OK" if r["on_topic"] == exp_top and r["committal"] == exp_com else "FAIL"
        print(f"  [{ok}] on_topic={r['on_topic']:<5} committal={r['committal']:<5}  "
              f"(expected {exp_top}, {exp_com})")
        print(f"    {text}")
