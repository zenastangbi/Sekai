"""Locator prompts, input batches, and box parsing."""

import copy
import json
from types import SimpleNamespace

END = (
    "Current chunk ends here. Return only the raw JSON object with exactly "
    "one field, bbox_2d, for the last frame."
)


def fresh_prompt(row, call, new_memory):
    """Return a prompt from row, call, and new_memory."""
    query = row["raw_query"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Original query is required")
    if not isinstance(new_memory, str):
        raise TypeError("New memory must be text")
    ids = call["real_indices"]
    if (
        not ids
        or ids != list(range(ids[0], ids[-1] + 1))
        or ids[-1] != call["target_index"]
    ):
        raise ValueError(
            "Only a contiguous arrived chunk ending at the target is allowed"
        )
    if len(ids) > 8 or not len(ids) == len(call["paths"]) == len(
        call["source_indices"]
    ):
        raise ValueError("Real image and source identities must agree")

    return (
        "Locate every visible instance in the LAST frame that satisfies the "
        "original query.\n\nUse the supplied memory together with the current "
        "frames to identify the correct instance(s) and their current "
        "locations.\n\nReturn exactly one raw JSON "
        'object:\n{{"bbox_2d":[[x1,y1,x2,y2], ...]}}\n`bbox_2d` must always '
        "be a list of boxes, even when there is only one target.\nFor no "
        'target: {{"bbox_2d":[]}}\n\nUse 0..1000 normalized xyxy '
        "coordinates. Output one box for each matching visible "
        "instance.\n\nOriginal query:\n{query}\n\nCurrent frame "
        "indices:\n{indices}\n\nMemory:\n{new_memory}"
    ).format(query=query, indices=json.dumps(ids), new_memory=new_memory)


class Processor:
    """Processor proxy with locator message text."""

    def __init__(self, original, prompt):
        """Store the wrapped processor and locator prompt."""
        self.original, self.prompt = original, prompt

    def __getattr__(self, name):
        """Return the named attribute from the wrapped processor."""
        return getattr(self.original, name)

    def __call__(self, *args, **kwargs):
        """Return the wrapped processor output for the arguments."""
        return self.original(*args, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        """Return rendered messages with locator text inserted."""
        messages = copy.deepcopy(messages)
        messages[0]["content"][0]["text"] = self.prompt
        messages[0]["content"][-1]["text"] = END
        self.messages = messages
        return self.original.apply_chat_template(messages, **kwargs)


def fresh_bbox_batch(route, row, call, new_memory):
    """Encode the query, call frames, and new_memory with route.

    Return model inputs, prefix length, and input metadata.
    """
    from multimodal import batch_for

    text = fresh_prompt(row, call, new_memory)
    tokens = len(
        route.processor.tokenizer.encode(new_memory, add_special_tokens=False)
    )
    if tokens > 128:
        raise ValueError(f"New memory exceeds the fixed capacity: {tokens}")
    proxy = Processor(route.processor, text)
    view = SimpleNamespace(qwen=route.qwen, processor=proxy)
    chunk = dict(
        paths=call["paths"],
        observation_ids=call["real_indices"],
        source_indices=call["source_indices"],
        times_seconds=call["real_indices"],
        units="INDEX",
    )

    data, prefix, audit = batch_for(view, chunk)
    assert "past_key_values" not in data
    audit.update(
        real_indices=call["real_indices"],
        target_index=call["target_index"],
        new_memory=new_memory,
        new_memory_tokens=tokens,
        previous_memory_in_input=False,
        old_memory_direct_input=False,
        GT_in_input=False,
        language_KV_reused=False,
        memory_semantically_may_retain_history=True,
        prompt=text,
        messages=proxy.messages,
        rendered_prompt=route.processor.tokenizer.decode(
            data["input_ids"][0, :prefix], skip_special_tokens=False
        ),
    )
    return data, prefix, audit


def parse_bbox(raw, truncated=False):
    """Return a validation record from raw box JSON.

    Return normalized boxes, empty-output status, and format errors.
    """
    import math

    errors = []

    def unique(pairs):
        """Build a dict from pairs; reject duplicate keys."""
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError("DUPLICATE_KEY")
            d[k] = v
        return d

    boxes = []
    try:
        obj = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(obj, dict) or set(obj) != {"bbox_2d"}:
            raise ValueError("EXACT_SINGLE_BBOX_2D_FIELD_REQUIRED")
        if not isinstance(obj["bbox_2d"], list):
            raise ValueError("BOX_LIST_REQUIRED")
        for v in obj["bbox_2d"]:
            if not isinstance(v, list) or len(v) != 4:
                raise ValueError("FOUR_COORDINATES_REQUIRED")
            if not all(
                isinstance(n, (int, float))
                and not isinstance(n, bool)
                and math.isfinite(n)
                and 0 <= n <= 1000
                for n in v
            ):
                raise ValueError("COORDINATE_RANGE_OR_TYPE")
            if not (v[0] < v[2] and v[1] < v[3]):
                raise ValueError("XYXY_ORDER")
            boxes.append([n / 1000 for n in v])
    except (ValueError, TypeError) as e:
        errors.append(str(e))
    if truncated:
        errors.append("TRUNCATED")
    valid = not errors
    return dict(
        raw=raw,
        valid=valid,
        full_prompt_valid=valid,
        canonical_bbox_schema=valid,
        format_failure=not valid,
        boxes=boxes if valid else [],
        legal_empty=valid and not boxes,
        errors=errors,
        schema_error=errors[0] if errors else None,
        truncated=truncated,
        coordinate_range=[0, 1000],
        clipped=False,
    )
