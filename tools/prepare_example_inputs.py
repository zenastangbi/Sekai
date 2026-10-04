"""Inference and scoring manifest construction."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
ROOT = Path(__file__).resolve().parents[1]


def build_inputs(directory, assets):
    """Build manifests from directory and the assets root.

    Return inference rows, scoring points, and excluded points.
    """
    from common import validate_rows

    directory, assets = Path(directory), Path(assets).resolve()

    def read(path):
        """Return the JSON value stored at path."""
        return json.loads(path.read_text(encoding="utf8"))

    index = read(directory / "index.json")
    classification_path = directory / "scoring_classification.json"
    classes = read(classification_path) if classification_path.exists() else []
    by_label = {r["label_id"]: r for r in classes}
    if len(by_label) != len(classes):
        raise ValueError("Duplicate classification point")
    rows, points, excluded = [], [], []
    for spec in index:
        name = spec["video_id"]
        z = read(directory / (name + ".json"))
        frames = z["video"]["frames"]
        for q in z["queries"]:
            rows.append(
                dict(
                    id=q["query_id"],
                    source_video=name,
                    expression_id=q["query_id"],
                    raw_query=q["english"],
                    frames=[str(assets / f["image"]) for f in frames],
                    frame_names=[
                        f"{f['published_index']:06d}" for f in frames
                    ],
                    source_indices=list(range(len(frames))),
                )
            )
        for r in z["labels"]:
            if not r["score"]:
                excluded.append(
                    dict(
                        video=name,
                        query_id=r["query_id"],
                        frame=r["published_index"],
                        score=False,
                        reason="Original source score=false",
                    )
                )
                continue
            f = frames[r["published_index"]]
            point = dict(
                query_id=r["query_id"],
                video=name,
                frame=r["published_index"],
                shape=[f["height"], f["width"]],
                gt_masks=[g["mask"] for g in r["gt"]],
                mode=r["mode"],
                target_count="0"
                if not r["output_ids"]
                else "1"
                if len(r["output_ids"]) == 1
                else "2+",
                history=None,
                age_bin=None,
                label_id=r["label_id"],
                is_p8_endpoint=r["is_p8_endpoint"],
            )
            if classes:
                c = by_label[r["label_id"]]
                for key in [
                    "video",
                    "query_id",
                    "frame",
                    "mode",
                    "target_count",
                ]:
                    if c[key] != point[key]:
                        raise ValueError(
                            "Classification/source mismatch: " + r["label_id"]
                        )
                if c["history"] is not None and c["mode"] not in [
                    "NOW",
                    "EVER",
                    "NEVER",
                ]:
                    raise ValueError("Unsupported history family")
                for key in [
                    "history",
                    "age",
                    "age_bin",
                    "age_basis",
                    "tau",
                    "chunk_start",
                    "window_reference",
                ]:
                    if key in c:
                        point[key] = c[key]
            points.append(point)
    if classes and {p["label_id"] for p in points} != set(by_label):
        raise ValueError("Classification must cover exactly the scored subset")
    if len({(r["query_id"], r["frame"]) for r in points}) != len(points):
        raise ValueError("Duplicate scoring point")
    validate_rows(rows)
    return rows, points, excluded


def write_inputs(output, rows, points):
    """Write rows and points to a new output directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [
        ("inference_manifest.json", rows),
        ("scoring_points.json", points),
    ]:
        with (output / name).open("x", encoding="utf8") as f:
            json.dump(value, f, indent=2)


def main():
    """Write manifests from CLI input paths and print their counts."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--examples", default=str(ROOT / "examples"))
    a = p.parse_args()
    rows, points, excluded = build_inputs(a.examples, a.assets)
    write_inputs(a.output, rows, points)
    print(
        json.dumps(
            dict(
                queries=len(rows),
                scoring_points=len(points),
                source_excluded=len(excluded),
                scope=(
                    "seven-video examples; use evaluate_examples.py for "
                    "preparation and scoring"
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
