"""Writer token generation and position-cache handling."""


def reset_position_cache(route):
    """Set cached rope_deltas on route model components to None."""
    for module in (route.qwen, getattr(route.qwen, "model", None)):
        if module is not None and hasattr(module, "rope_deltas"):
            module.rope_deltas = None


def generate(route, batch, seed, max_new_tokens):
    """Return writer text and metadata from batch and seed.

    Return decoded text, token IDs, termination flags, and elapsed
    time.
    """
    import time

    import torch

    route.eval()
    reset_position_cache(route)
    assert "labels" not in batch and "past_key_values" not in batch
    start = time.perf_counter()

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            kwargs = dict(
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
            output = route.qwen.generate(**batch, **kwargs)
    prefix_length = batch["input_ids"].shape[1]
    ids = output[0, prefix_length:].detach().cpu()
    eos = route.qwen.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    ended = bool(ids.numel() and int(ids[-1]) in eos)
    raw = route.processor.tokenizer.decode(ids, skip_special_tokens=True)
    return dict(
        raw=raw,
        generated_token_ids=ids.tolist(),
        seed=seed,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        truncated=not ended,
        ended_with_eos=ended,
        seconds=time.perf_counter() - start,
        semantic_retries=0,
    )
