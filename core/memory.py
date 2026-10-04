"""Event-memory prompts, serialization, and parsing."""

import json
import re

CAPACITY = 128
STATES = {"active", "ended", "unknown"}
WRITER_INSTRUCTIONS = (
    "Update object memory from ONLY the supplied arrived images and previous "
    "memory.\nThe query defines what matters; it is NOT evidence that an "
    "event happened.\nKeep stable object identities across motion; never "
    "transfer history to a different object.\nRecord recognizable appearance, "
    "last actually visible location and frame INDEX, and observed relevant "
    "events.\nFor each event: active means visibly ongoing at the latest "
    "observation; ended requires observed evidence of ending; unknown means "
    "its current state cannot be determined. Disappearance alone does not "
    "establish ending. An event's last evidence frame is NOT its end time. Do "
    "not invent events, infer hidden causes, or disguise guesses with "
    "possibly.\nWrite a COMPLETE replacement, preserving useful supported "
    "history and explicit uncertainty.\nReturn ONLY "
    '<memory>{"objects":[["o1","appearance","last visible '
    'location",8,[["neutral event label","active",0,8]]]]}</memory>.\nEac'
    "h object is [stable local ID, appearance, last visible location, last "
    "visible frame INDEX, events]. Each event is [neutral event label, "
    "active/ended/unknown, first evidence INDEX, last evidence INDEX]. Event "
    "labels name actions, not whether they started or ended. Use integers "
    "from frames actually observed; never seconds. Empty events and "
    '{"objects":[]} are legal.\nThe JSON body AND its concise rendered text '
    "must fit 128 tokens. Prefer a few short, useful observations. No extra "
    "fields, coordinates, GT labels, or text outside the memory tags."
)


def dumps(value):
    """Serialize value as compact Unicode JSON."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def writer_prompt(query, indices, previous_json=""):
    """Build a prompt from query, indices, and previous_json."""
    if not indices or any(b != a + 1 for a, b in zip(indices, indices[1:])):
        raise ValueError("Non-contiguous real input")
    return (
        WRITER_INSTRUCTIONS
        + "\n\nOriginal query:\n"
        + query
        + "\n\nPrevious memory:\n"
        + (previous_json or '{"objects":[]}')
        + "\n\nArrived frame INDEX values:\n"
        + dumps(indices)
    )


def render(memory):
    """Return locator text from the object records in memory."""
    lines = []
    for identity, look, location, seen, events in memory["objects"]:
        facts = []
        for event, status, first, last in events:
            state = " " + status
            facts.append(f"{event}{state}; evidence {first}-{last}")
        lines.append(
            f"{identity}: {look}; last seen {seen} {location}"
            + (
                "; " + "; ".join(facts)
                if facts
                else "; no confirmed relevant event"
            )
        )
    return "\n".join(lines)


def failed_update(previous_json=""):
    """Return a failure record retaining previous_json."""
    structured = (
        json.loads(previous_json) if previous_json else {"objects": []}
    )
    return {
        "valid": False,
        "writer_memory_out": previous_json,
        "structured": structured,
        "memory_out": render(structured),
        "run_locator": False,
        "locator_call_eligible": False,
        "emit_new_bbox": False,
    }


def parse(raw, *, tokenizer, observed_indices):
    """Parse raw memory text using tokenizer and observed_indices.

    Return object records, canonical JSON, rendered text, and token
    counts. Raise ValueError for invalid fields or excess tokens.
    """
    match = re.fullmatch(r"\s*<memory>\s*(.*?)\s*</memory>\s*", raw, re.S)
    if not match:
        raise ValueError("MEMORY_TAGS")

    def unique(items):
        """Build a dict from items; reject duplicate keys."""
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError("DUPLICATE_JSON_KEY")
            out[k] = v
        return out

    obj = json.loads(match.group(1), object_pairs_hook=unique)
    if (
        not isinstance(obj, dict)
        or set(obj) != {"objects"}
        or not isinstance(obj["objects"], list)
    ):
        raise ValueError("OBJECT_SCHEMA")
    allowed = set(observed_indices)
    ids = set()
    for row in obj["objects"]:
        if not isinstance(row, list) or len(row) != 5:
            raise ValueError("OBJECT_ROW")
        identity, look, location, seen, events = row
        if (
            not isinstance(identity, str)
            or not re.fullmatch(r"o[1-9][0-9]*", identity)
            or identity in ids
        ):
            raise ValueError("LOCAL_ID")
        ids.add(identity)
        if any(
            not isinstance(v, str) or not v.strip() for v in [look, location]
        ):
            raise ValueError("DESCRIPTION")
        if type(seen) is not int or seen not in allowed:
            raise ValueError("UNOBSERVED_LAST_SEEN")
        if not isinstance(events, list):
            raise ValueError("EVENTS")
        for e in events:
            if not isinstance(e, list) or len(e) != 4:
                raise ValueError("EVENT_ROW")
            event, state, first, last = e
            if (
                not isinstance(event, str)
                or not event.strip()
                or not isinstance(state, str)
                or state not in STATES
            ):
                raise ValueError("EVENT_STATE")
            if (
                type(first) is not int
                or type(last) is not int
                or first not in allowed
                or last not in allowed
                or first > last
            ):
                raise ValueError("UNOBSERVED_EVENT_TIME")
    canonical = dumps(obj)
    text = render(obj)
    body_tokens = len(tokenizer.encode(canonical, add_special_tokens=False))
    rendered_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if max(body_tokens, rendered_tokens) > CAPACITY:
        raise ValueError("MEMORY_OVER_128")
    return {
        "structured": obj,
        "canonical": canonical,
        "rendered": text,
        "body_tokens": body_tokens,
        "rendered_tokens": rendered_tokens,
        "semantic_truth_verified": False,
    }
