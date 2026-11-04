"""JSON box grammar and constrained token generation."""

import functools
import time

import regex

W = r"[ \t\r\n]*"
N = r"(?:1000(?:\.0+)?|(?:0|[1-9][0-9]{0,2})(?:\.[0-9]+)?)"
B = r"\[" + W + N + (W + "," + W + N) * 3 + W + r"\]"
PAT = (
    r"\{"
    + W
    + r'"bbox_2d"'
    + W
    + ":"
    + W
    + r"\["
    + W
    + "(?:"
    + B
    + "(?:"
    + W
    + ","
    + W
    + B
    + ")*)?"
    + W
    + r"\]"
    + W
    + r"\}"
)
GRAMMAR = regex.compile(PAT)
SCHEMA = {
    "type": "object",
    "properties": {
        "bbox_2d": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1000},
                "minItems": 4,
                "maxItems": 4,
            },
        }
    },
    "required": ["bbox_2d"],
    "additionalProperties": False,
}


class Constraint:
    """Token-prefix validator for the box JSON grammar."""

    def __init__(self, tokenizer):
        """Store tokenizer and candidate tokens for JSON boxes."""
        self.tokenizer = tokenizer
        allowed = set('{}[]"bbox_2d:0123456789., \t\r\n')
        texts = tokenizer.batch_decode(
            [[i] for i in range(len(tokenizer))],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        self.candidates = [
            (i, t)
            for i, t in enumerate(texts)
            if t and set(t) <= allowed and i not in tokenizer.all_special_ids
        ]

    @functools.lru_cache(maxsize=16384)
    def allowed(self, text):
        """Return allowed next-token IDs for text."""
        ids = [
            i
            for i, t in self.candidates
            if GRAMMAR.fullmatch(text + t, partial=True) is not None
        ]
        if not ids:
            raise RuntimeError(
                "JSON grammar has no allowed token at " + repr(text[-100:])
            )
        return ids

    def text(self, ids):
        """Decode ids with special tokens and whitespace preserved."""
        return self.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def complete(self, text):
        """Return whether text matches a complete box JSON object."""
        return GRAMMAR.fullmatch(text) is not None


def generate_constrained(route, batch, seed, limit):
    """Generate at most limit tokens from route and batch using seed.

    Return raw text, parsed boxes, token IDs, and decoding metadata.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    from generation import reset_position_cache
    from locator import parse_bbox

    route.eval()
    reset_position_cache(route)
    if not hasattr(route, "_bbox_constraint"):
        route._bbox_constraint = Constraint(route.processor.tokenizer)
    c = route._bbox_constraint
    prefix = batch["input_ids"].shape[1]

    def allowed(batch_id, ids):
        """Return allowed next-token IDs for the input prefix."""
        return c.allowed(c.text(ids[prefix:].tolist()))

    class Stop(StoppingCriteria):
        """Stopping criterion for complete box JSON responses."""

        def __call__(self, input_ids, scores, **kwargs):
            """Return completion flags for input_ids."""
            return torch.tensor(
                [
                    c.complete(c.text(row[prefix:].tolist()))
                    for row in input_ids
                ],
                device=input_ids.device,
                dtype=torch.bool,
            )

    start = time.perf_counter()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = route.qwen.generate(
                **batch,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                max_new_tokens=limit,
                use_cache=True,
                prefix_allowed_tokens_fn=allowed,
                stopping_criteria=StoppingCriteriaList([Stop()]),
            )
    torch.cuda.synchronize()
    ids = output[0, prefix:].tolist()
    raw = c.text(ids)
    closed = c.complete(raw)
    return dict(
        raw=raw,
        generated_token_ids=ids,
        parsed=parse_bbox(raw, not closed),
        seed=seed,
        do_sample=False,
        truncated=not closed,
        ended_with_eos=False,
        stopped_on_top_level_json=closed,
        seconds=time.perf_counter() - start,
        semantic_retries=0,
        constrained_decoding=True,
        schema=SCHEMA,
        grammar=PAT,
    )
