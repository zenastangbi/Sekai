"""Qwen base-model and processor loading."""


def load(cfg):
    """Return a frozen model and processor from cfg."""
    import torch
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
    assert not any("lora" in n.lower() for n, _ in q.named_parameters())
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
    for p in q.parameters():
        p.requires_grad_(False)
    q.eval()
    q.to("cuda")
    return Route(q, processor)
