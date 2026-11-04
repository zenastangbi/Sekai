"""Frame-by-frame mask propagation and storage."""

import contextlib
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

FIRST_OBJECT_ID = 17
OBJECT_ID_STEP = 3


class TrackingExecutor:
    """Segmentation state for one video-query sequence."""

    def __init__(self, model, scratch):
        """Store model and scratch and initialize tracking state."""
        self.model = model
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.anchor = None
        self.boxes = []
        self.last_tick = -1
        self.query_id = None

    def step(self, row, tick, prediction=None):
        """Process row at tick with an optional locator prediction.

        Return a union mask, per-object masks, and frame execution
        metadata.
        """
        import torch

        from sam_masks import _union_output_masks

        assert tick == self.last_tick + 1
        if self.query_id is None:
            self.query_id = row["id"]
        assert self.query_id == row["id"], (
            "Cross-query tracking state contamination"
        )
        self.last_tick = tick
        if prediction is not None:
            assert (
                prediction["request"]["anchor_source_id"]["published_tick"]
                == tick
            )
            assert (
                prediction["request"]["anchor_source_id"]["video"]
                == row["source_video"]
            )
            if not prediction["format_failure"]:
                self.anchor = tick
                self.boxes = prediction["boxes"]
        with Image.open(row["frames"][tick]) as image:
            shape = (image.height, image.width)
        union = np.zeros(shape, bool)
        instance_masks = []
        event = dict(
            tick=tick,
            segment_anchor=self.anchor,
            boxes=self.boxes,
            max_source_tick_accessed=tick,
            execution=("forward propagation over the arrived segment"),
            failure_fallback=bool(prediction and prediction["format_failure"]),
            sam_packets=0,
            sam_prompt_seconds=0.0,
            sam_propagation_seconds=0.0,
            io_and_session_seconds=0.0,
        )
        begin = time.perf_counter()
        if not self.boxes:
            event["total_seconds"] = time.perf_counter() - begin
            return union, instance_masks, event
        assert self.anchor is not None and self.anchor <= tick

        with contextlib.nullcontext(
            tempfile.mkdtemp(prefix="arrived_", dir=self.scratch)
        ) as directory:
            for offset, index in enumerate(range(self.anchor, tick + 1)):
                Path(directory, f"{offset:06d}.jpg").symlink_to(
                    row["frames"][index]
                )
            count = tick - self.anchor + 1
            for j, b in enumerate(self.boxes):
                t0 = time.perf_counter()
                session = self.model.handle_request(
                    dict(
                        type="start_session",
                        resource_path=directory,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=False,
                    )
                )["session_id"]
                torch.cuda.synchronize()
                event["io_and_session_seconds"] += time.perf_counter() - t0
                object_id = FIRST_OBJECT_ID + j * OBJECT_ID_STEP

                def mask_of(packet):
                    """Return a boolean union mask from packet."""
                    ids = (
                        np.asarray(packet["outputs"]["out_obj_ids"])
                        .reshape(-1)
                        .tolist()
                    )
                    assert set(ids).issubset({object_id})
                    m = _union_output_masks(packet["outputs"])
                    result = (
                        np.zeros(shape, bool)
                        if m is None
                        else m.numpy().astype(bool).copy()
                    )
                    assert result.shape == shape
                    return result

                try:
                    t0 = time.perf_counter()
                    prompted = self.model.handle_request(
                        dict(
                            type="add_prompt",
                            session_id=session,
                            frame_index=0,
                            points=[[b[0], b[1]], [b[2], b[3]]],
                            point_labels=[2, 3],
                            obj_id=object_id,
                            rel_coordinates=True,
                        )
                    )
                    assert int(prompted["frame_index"]) == 0
                    current = mask_of(prompted)
                    torch.cuda.synchronize()
                    event["sam_prompt_seconds"] += time.perf_counter() - t0
                    if count > 1:
                        t0 = time.perf_counter()
                        seen = set()
                        for packet in self.model.propagate_in_video(
                            session_id=session,
                            propagation_direction="forward",
                            start_frame_idx=0,
                            max_frame_num_to_track=count,
                            is_last_batch=True,
                        ):
                            index = int(packet["frame_index"])
                            assert 0 <= index < count and index not in seen
                            seen.add(index)
                            event["sam_packets"] += 1
                            if index == count - 1:
                                current = mask_of(packet)
                        assert count - 1 in seen, "Missing current frame"
                        torch.cuda.synchronize()
                        event["sam_propagation_seconds"] += (
                            time.perf_counter() - t0
                        )
                    instance_masks.append(current.copy())
                    union |= current
                finally:
                    t0 = time.perf_counter()
                    self.model.handle_request(
                        dict(
                            type="close_session",
                            session_id=session,
                            run_gc_collect=True,
                        )
                    )
                    torch.cuda.synchronize()
                    event["io_and_session_seconds"] += time.perf_counter() - t0
        event["total_seconds"] = time.perf_counter() - begin
        return union, instance_masks, event


def build_model(root):
    """Return a SAM predictor using root and SEKAI_SAM_CHECKPOINT."""
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference")
    model = build_sam3_multiplex_video_predictor(
        checkpoint_path=os.environ["SEKAI_SAM_CHECKPOINT"],
        bpe_path=str(Path(root) / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
        strict_causal=True,
        strict_research_checkpoint=True,
    )
    return model


def commit_mask(path, mask):
    """Write a boolean mask to path as a PNG.

    If path exists, return None for an equal mask or raise
    AssertionError.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        assert np.array_equal(np.asarray(Image.open(path)) > 0, mask), (
            "Attempt to overwrite past prediction"
        )
        return
    with path.open("xb") as handle:
        Image.fromarray(mask.astype("uint8") * 255).save(handle, format="PNG")
