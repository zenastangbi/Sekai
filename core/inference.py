"""Locator and segmentor command-line execution."""

import argparse
import os
import sys
from pathlib import Path

from common import bind, chunks, dump, read, validate_rows


def reject_gt(event, args):
    """Reject open events targeting mask_dict.json."""
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).replace("\\", "/").split("/")[-1]
        if name == "mask_dict.json":
            raise RuntimeError(
                "Ground truth access prohibited in official inference"
            )


def main():
    """Write locator or mask outputs from CLI inputs."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--phase", choices=["locator", "sam"], required=True)
    a = p.parse_args()
    sys.addaudithook(reject_gt)
    cfg = read(a.config)
    rows = read(a.manifest)
    validate_rows(rows)
    out = Path(a.output)
    identity = dict(config=cfg, manifest=rows, GT_read=False, method="sekai")
    bind(out / "binding.json", identity)
    import torch

    from locator import fresh_bbox_batch

    if a.phase == "locator":
        route = None
        for row in rows:
            base = out / "queries" / row["source_video"] / row["expression_id"]
            for c in chunks(row):
                name = f"{c['call']:04d}.json"
                dest = base / a.phase / name
                cache_path = (
                    Path(cfg["writer_cache"])
                    / "queries"
                    / row["source_video"]
                    / row["expression_id"]
                    / "writer"
                    / name
                )

                m = read(cache_path)
                selected_memory = m["memory_out"]
                assert m["call"] == c
                if dest.exists():
                    z = read(dest)
                    assert z["call"] == c and z["writer_record"] == m
                else:
                    z = dict(
                        call=c,
                        writer_record=m,
                        normal=dict(boxes=[], format_failure=True),
                        blocked=not m["valid"],
                        writer_update_valid=m["valid"],
                        locator_called=False,
                    )
                    if m["valid"]:
                        if route is None:
                            from locator_model import load as load_lora

                            route = load_lora(
                                cfg["locator"], cfg["checkpoint"]
                            )
                        from constrained import generate_constrained

                        batch, _, audit = fresh_bbox_batch(
                            route, row, c, selected_memory
                        )
                        assert not audit["GT_in_input"] and (
                            not audit["language_KV_reused"]
                        )
                        g = generate_constrained(
                            route, batch, cfg["writer"]["seed"], 256
                        )
                        del batch
                        z.update(
                            locator_called=True,
                            input=audit,
                            generation=g,
                            normal=dict(
                                boxes=g["parsed"]["boxes"],
                                format_failure=not g["parsed"]["valid"],
                            ),
                        )
                    dump(dest, z)
                dump(
                    out / "progress.json",
                    dict(phase=a.phase, query=row["id"], call=c["call"]),
                )
        dump(
            out / (a.phase + "_complete.json"),
            dict(queries=len(rows), **identity),
        )
        return
    from sam_adapter import TensorPromptBackend
    from segmentor import TrackingExecutor, build_model, commit_mask

    backend = TensorPromptBackend(build_model(cfg["sam_source"]))
    for row in rows:
        base = out / "queries" / row["source_video"] / row["expression_id"]
        anchors = {
            c["target_index"]: read(base / "locator" / f"{c['call']:04d}.json")
            for c in chunks(row)
        }
        executor = TrackingExecutor(backend, out / "scratch")
        resume = base / "sam_resume.json"
        state = (
            read(resume) if resume.exists() else dict(next_tick=0, segment={})
        )
        for key, value in state["segment"].items():
            setattr(executor, key, value)
        safe = {key: row[key] for key in ["id", "source_video", "frames"]}
        for tick in range(state["next_tick"], len(row["frames"])):
            pred = None
            if tick in anchors:
                n = anchors[tick]["normal"]
                pred = dict(
                    **n,
                    request=dict(
                        anchor_source_id=dict(
                            published_tick=tick, video=row["source_video"]
                        )
                    ),
                )
            with (
                torch.inference_mode(),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                mask, instances, event = executor.step(safe, tick, pred)
            prior_legal = state.get("has_legal_selection", False)
            valid_update = pred is not None and (not pred["format_failure"])
            event.update(
                writer_update_valid=anchors[tick].get("writer_update_valid")
                if tick in anchors
                else None,
                locator_called=anchors[tick].get("locator_called", False)
                if tick in anchors
                else False,
                selection_reused=bool(not valid_update and prior_legal),
                no_legal_selection_yet=bool(
                    not prior_legal and (not valid_update)
                ),
            )
            has_legal_selection = prior_legal or valid_update
            import numpy as np

            ip = base / "instances" / f"{tick:06d}.npz"
            ip.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                ip,
                masks=np.stack(instances)
                if instances
                else np.zeros((0, *mask.shape), dtype=bool),
            )
            commit_mask(
                out
                / "predictions"
                / row["source_video"]
                / row["expression_id"]
                / (row["frame_names"][tick] + ".png"),
                mask,
            )
            dump(base / "events" / f"{tick:06d}.json", event)
            state = dict(
                next_tick=tick + 1,
                has_legal_selection=has_legal_selection,
                segment={
                    key: getattr(executor, key)
                    for key in ["anchor", "boxes", "last_tick", "query_id"]
                },
            )
            dump(resume, state)
            dump(
                out / "progress.json",
                dict(phase="sam", query=row["id"], tick=tick),
            )
        dump(
            base / "complete.json", dict(frames=len(row["frames"]), **identity)
        )
    dump(
        out / "complete.json",
        dict(
            status="INFERRED_NOT_SCORED",
            queries=len(rows),
            frames=sum((len(r["frames"]) for r in rows)),
            **identity,
        ),
    )


if __name__ == "__main__":
    main()
