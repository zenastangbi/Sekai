"""Isekai mask metrics and video-level aggregation."""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image

import metrics
from metric_contract import jf


def aggregate(rows):
    """Return video-equal J, F, and JF means from rows."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["video"]].append(row)
    return {
        key: sum(
            sum(float(r[key]) for r in v) / len(v) for v in groups.values()
        )
        / len(groups)
        if groups
        else None
        for key in ["J", "F", "JF"]
    }


def summarize(rows):
    """Return subgroup metric means and counts from rows."""
    groups = {"all": rows}
    for field in ["mode", "target_count", "age_bin"]:
        for value in sorted(
            {str(r[field]) for r in rows if r.get(field) not in [None, ""]}
        ):
            groups[field + ":" + value] = [
                r for r in rows if str(r.get(field)) == value
            ]
    groups["Other"] = [
        r
        for r in rows
        if r.get("mode") not in ["NOW", "EVER", "LATEST", "NEVER"]
    ]
    groups["history"] = [
        r for r in rows if r.get("history") in [True, "True", "true", "1"]
    ]
    return {
        name: dict(
            records=len(rs),
            videos=len({r["video"] for r in rs}),
            **aggregate(rs),
        )
        for name, rs in groups.items()
    }


def binary(path, shape):
    """Return the boolean mask at path with the given shape."""
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim != 2 or a.shape != tuple(shape):
        raise ValueError("Invalid union-mask dimensions")
    return a > 0


def score_point(point, prediction, data_root, prediction_root):
    """Score prediction against point using the GT and prediction roots.

    Return point metadata, J/F/JF scores, and prediction status.
    """
    shape = point["shape"]
    target = np.zeros(shape, bool)
    for mask in point["gt_masks"]:
        target |= binary(Path(data_root) / mask, shape)
    row = {k: v for k, v in point.items() if k not in ["shape", "gt_masks"]}
    status = "valid"
    error = None
    try:
        if prediction is None:
            raise FileNotFoundError("No prediction manifest entry")
        if prediction.get("invalid"):
            raise ValueError("Explicit invalid output")
        masks = prediction["masks"]
        if not isinstance(masks, list):
            raise ValueError("masks must be a list, including []")
        pred = np.zeros(shape, bool)
        for path in masks:
            pred |= binary(Path(prediction_root) / path, shape)
        j, f, s = jf(target, pred, metrics)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        status = "missing" if isinstance(exc, FileNotFoundError) else "invalid"
        j = f = s = 0.0
        error = str(exc)
    row.update(
        J=j,
        F=f,
        JF=s,
        prediction_status=status,
        failure=int(status != "valid"),
        error=error,
    )
    return row


def load_json(path):
    """Return JSON data from path, accepting an optional UTF-8 BOM."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def key(r):
    """Return the query ID and integer frame index of record r."""
    return r["query_id"], int(r["frame"])


def main():
    """Write scores, coverage, and group means from CLI inputs."""
    p = argparse.ArgumentParser()
    p.add_argument("--points", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--prediction-root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    points = load_json(a.points)
    preds = load_json(a.predictions)
    if len({key(x) for x in points}) != len(points) or len(
        {key(x) for x in preds}
    ) != len(preds):
        raise ValueError("Duplicate scoring/prediction key")
    by = {key(x): x for x in preds}
    extra = set(by) - {key(x) for x in points}
    if extra:
        raise ValueError("Unexpected predictions outside the scoring manifest")
    rows = [
        score_point(x, by.get(key(x)), a.data_root, a.prediction_root)
        for x in points
    ]
    counts = collections.Counter(r["prediction_status"] for r in rows)
    result = {
        "policy": "original_record_failure_zero",
        "records": len(rows),
        "coverage": counts["valid"] / len(rows) if rows else None,
        "missing": counts["missing"],
        "invalid": counts["invalid"],
        "groups": summarize(rows),
        "per_point": rows,
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["groups", "per_point"]
            }
        )
    )


if __name__ == "__main__":
    main()
