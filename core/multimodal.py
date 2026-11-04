"""Qwen image and video input preparation."""

import re

import torch


def batch_for(route, chunk):
    """Encode chunk image paths and frame metadata with route.

    Return model inputs, prefix length, and visual-token metadata.
    Odd video batches repeat their final frame for temporal padding.
    """
    import numpy as np
    from PIL import Image

    paths = list(chunk["paths"])
    sources = list(chunk["source_indices"])
    ticks = list(chunk.get("observation_ids", sources))
    times = list(chunk.get("times_seconds", ticks))
    if not paths or not (
        len(paths) == len(sources) == len(ticks) == len(times)
    ):
        raise ValueError(
            (
                "Paths, source identities, observation IDs and times must "
                "correspond"
            )
        )
    if len(set(ticks)) != len(ticks) or any(
        b <= a for a, b in zip(ticks, ticks[1:])
    ):
        raise ValueError(
            (
                "Real observations must be strictly chronological, without "
                "duplicates"
            )
        )
    ims = []
    for path in paths:
        with Image.open(path) as im:
            ims.append(im.convert("RGB"))
    isvideo = len(ims) > 1
    encoded_times = list(ticks if chunk.get("units") == "INDEX" else times)
    padding_map = [
        dict(
            input_index=i,
            observation_id=tick,
            source_index=sources[i],
            is_padding=False,
        )
        for i, tick in enumerate(ticks)
    ]
    if isvideo and len(ims) % 2:
        ims.append(ims[-1].copy())
        encoded_times.append(encoded_times[-1])
        padding_map.append(
            dict(
                input_index=len(ims) - 1,
                observation_id=ticks[-1],
                source_index=sources[-1],
                is_padding=True,
            )
        )
    processor = route.processor
    if isvideo:
        grid = processor.image_processor(
            images=[ims[-1]], return_tensors="pt"
        )["image_grid_thw"][0]
        patch = route.qwen.config.vision_config.patch_size
        height, width = int(grid[1]) * patch, int(grid[2]) * patch
        ims = [
            im.resize((width, height), Image.Resampling.BICUBIC) for im in ims
        ]

    text = route.processor.prompt
    ending = ""
    content = [
        {"type": "text", "text": text},
        {"type": "video" if isvideo else "image"},
        {"type": "text", "text": ending},
    ]
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    kwargs = dict(text=[rendered], return_tensors="pt", padding=False)
    if isvideo:
        kwargs.update(
            videos=[np.stack([np.asarray(im) for im in ims])],
            video_metadata=[
                dict(
                    total_num_frames=max(sources) + 1,
                    fps=1.0,
                    frames_indices=encoded_times,
                )
            ],
            do_sample_frames=False,
            do_resize=False,
        )
    else:
        kwargs["images"] = ims
    batch = dict(processor(**kwargs))
    if chunk.get("units") == "INDEX":
        decoded = processor.tokenizer.decode(
            batch["input_ids"][0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        decoded = re.sub(
            r"<([0-9.]+) seconds>", r"<observation INDEX \1>", decoded
        )
        tokens = processor.tokenizer(
            [decoded], return_tensors="pt", add_special_tokens=False
        )
        batch["input_ids"], batch["attention_mask"] = (
            tokens["input_ids"],
            tokens["attention_mask"],
        )
    visual_id = (
        route.qwen.config.video_token_id
        if isvideo
        else route.qwen.config.image_token_id
    )
    visual_positions = (
        (batch["input_ids"][0] == visual_id).nonzero().flatten().tolist()
    )
    if not visual_positions:
        raise ValueError("Native visual IDs missing")
    prefix = batch["input_ids"].shape[1]
    grid_key = "video_grid_thw" if isvideo else "image_grid_thw"
    grids = batch[grid_key].tolist()
    merge = route.qwen.config.vision_config.spatial_merge_size
    grid_height, grid_width = (
        int(grids[-1][1]) // merge,
        int(grids[-1][2]) // merge,
    )
    pixels = grid_height * grid_width
    if len(visual_positions) != sum(
        int(g[0]) * int(g[1]) * int(g[2]) // (merge * merge) for g in grids
    ):
        raise ValueError(
            "Native visual count does not agree with processor grid"
        )
    device = route.qwen.get_input_embeddings().weight.device
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    audit = dict(
        prefix_length=prefix,
        source_indices=sources,
        observation_ids=ticks,
        times_seconds=times,
        units=chunk.get("units", "INDEX"),
        processor_padding_map=padding_map,
        new_frame_ids=ticks,
        current_output_frame_id=ticks[-1],
        visual_grid=grids,
        spatial_grid_hw=[grid_height, grid_width],
        current_visual_positions=visual_positions[-pixels:],
        visual_tokens=len(visual_positions),
        modality="native_video" if isvideo else "image",
        prompt=text + ending,
        native_deepstack=True,
        native_mrope=True,
        semantic_kv_reused=False,
        input_ids=batch["input_ids"][0].tolist(),
        spatial_evidence=("last native temporal patch")
        if isvideo
        else "current image",
    )
    return batch, prefix, audit
