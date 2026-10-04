"""SAM object-mask decoding and union operations."""

from typing import Any, Mapping

import numpy as np
import torch


def _decode_outputs(outputs: Mapping[str, Any]) -> dict[int, torch.Tensor]:
    """Return object IDs mapped to CPU boolean masks."""
    raw_ids = outputs.get("out_obj_ids", ())
    raw_masks = outputs.get("out_binary_masks", ())
    ids = np.asarray(raw_ids).reshape(-1).tolist()
    masks = np.asarray(raw_masks)
    if masks.ndim == 2 and len(ids) == 1:
        masks = masks[None]
    if len(ids) != len(masks):
        raise RuntimeError("SAM3 output object and mask counts differ")
    return {
        int(object_id): torch.as_tensor(masks[index], dtype=torch.bool).cpu()
        for index, object_id in enumerate(ids)
    }


def _union_output_masks(outputs: Mapping[str, Any]) -> torch.Tensor | None:
    """Return the mask union from outputs, or None if empty."""
    decoded = list(_decode_outputs(outputs).values())
    if not decoded:
        return None
    return torch.stack(decoded, dim=0).any(dim=0)
