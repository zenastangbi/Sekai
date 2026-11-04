"""Locator checkpoint and processor loading."""

EXPECTED_LORA_TARGET_COUNT = 253


def load(cfg, checkpoint):
    """Return a locator and processor from cfg and checkpoint.

    The checkpoint supplies LoRA, merger, and normalization
    parameters.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from model import Route

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference")
    q = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    targets = [
        n
        for n, m in q.named_modules()
        if isinstance(m, torch.nn.Linear)
        and (
            (
                n.startswith("model.language_model.")
                and n.split(".")[-1]
                in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            )
            or n == "lm_head"
        )
    ]
    assert len(targets) == EXPECTED_LORA_TARGET_COUNT, (len(targets), targets)
    q = get_peft_model(
        q,
        LoraConfig(
            r=cfg["rank"],
            lora_alpha=cfg["alpha"],
            lora_dropout=cfg["dropout"],
            target_modules=targets,
            bias="none",
        ),
    )
    params = dict(q.named_parameters())
    loaded_names = {
        n
        for n in params
        if "lora_" in n
        or ("model.visual." in n and "merger" in n)
        or ("model.language_model." in n and "norm" in n)
    }
    for n in loaded_names:
        params[n].data = params[n].data.float()
    z = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if set(z["parameters"]) != loaded_names:
        raise ValueError(
            "Checkpoint parameter layout does not match this locator"
        )
    with torch.no_grad():
        for n, v in z["parameters"].items():
            params[n].copy_(v)
    for v in q.parameters():
        v.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(
        cfg["model"], local_files_only=True
    )
    processor.image_processor.max_pixels = cfg["pixels"]
    processor.image_processor.min_pixels = 4096
    processor.image_processor.size = {
        "shortest_edge": 4096,
        "longest_edge": cfg["pixels"],
    }
    if getattr(processor, "video_processor", None) is not None:
        processor.video_processor.size = dict(processor.image_processor.size)
    q.to("cuda")
    q.eval()
    return Route(q, processor)
