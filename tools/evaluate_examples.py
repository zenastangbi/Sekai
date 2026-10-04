"""Example input preparation and prediction scoring."""

import argparse
import collections
import datetime
import json
import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "tools"),
    str(Path(__file__).resolve().parents[1] / "core"),
    str(Path(__file__).resolve().parents[1] / "evaluation"),
]
ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
PROFILE = ROOT / "configs/examples.json"


def read(path):
    """Return JSON data from path, accepting an optional UTF-8 BOM."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    """Write value as UTF-8 JSON to a new file at path."""
    with Path(path).open("x", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def load_profile():
    """Return the configured example subset profile."""
    return read(PROFILE)


def predictions_for(points):
    """Return prediction-path records for scoring points."""
    return [
        dict(
            query_id=p["query_id"],
            frame=p["frame"],
            masks=[f"{p['video']}/{p['query_id']}/{p['frame']:06d}.png"],
        )
        for p in points
    ]


def prepare(assets, output):
    """Validate assets and write example input manifests to output.

    Return subset counts and preparation metadata.
    """
    from check_examples import check
    from common import chunks
    from prepare_example_inputs import build_inputs, write_inputs

    profile = load_profile()
    if Path(output).exists():
        raise FileExistsError("Use a new input directory")
    check(EXAMPLES, assets)
    rows, points, excluded = build_inputs(EXAMPLES, assets)
    if [r["id"] for r in rows] != profile["queries"] or len(points) != profile[
        "counts"
    ]["scoring_points"]:
        raise ValueError("Selected subset does not match its profile")
    write_inputs(output, rows, points)
    output = Path(output)
    write(output / "predictions_manifest.json", predictions_for(points))
    write(output / "source_excluded_points.json", excluded)
    binding = dict(
        schema="isekai-example-inputs-v1",
        subset_id=profile["subset_id"],
        scope="example_subset_only",
        full_isekai_result=False,
        counts=profile["counts"],
        all_query_frame_outputs=sum(len(r["frames"]) for r in rows),
        excluded_by_original_score_flag=len(excluded),
        writer_calls=sum(len(chunks(r)) for r in rows),
        assets_root=str(Path(assets).resolve()),
        assets_present=True,
        files=[
            "inference_manifest.json",
            "scoring_points.json",
            "predictions_manifest.json",
            "source_excluded_points.json",
        ],
    )
    write(output / "subset.json", binding)
    return {
        k: v for k, v in binding.items() if k not in ["files", "assets_root"]
    }


def load_inputs(directory):
    """Return validated manifests from directory.

    Return profile, binding, inference rows, and scoring points.
    """
    from prepare_example_inputs import build_inputs

    directory = Path(directory)
    profile, binding = load_profile(), read(directory / "subset.json")
    if binding["subset_id"] != profile["subset_id"]:
        raise ValueError("Input binding belongs to another subset/profile")
    required = {
        "inference_manifest.json",
        "scoring_points.json",
        "predictions_manifest.json",
        "source_excluded_points.json",
    }
    if set(binding["files"]) != required:
        raise ValueError("Incomplete input binding")
    rows, points, excluded = build_inputs(EXAMPLES, binding["assets_root"])
    if (
        binding.get("scope") != "example_subset_only"
        or binding.get("full_isekai_result") is not False
        or binding.get("counts") != profile["counts"]
        or binding.get("excluded_by_original_score_flag") != len(excluded)
        or binding.get("all_query_frame_outputs")
        != sum(len(r["frames"]) for r in rows)
    ):
        raise ValueError("Subset scope/counts changed")
    expected = {
        "inference_manifest.json": rows,
        "scoring_points.json": points,
        "predictions_manifest.json": predictions_for(points),
        "source_excluded_points.json": excluded,
    }
    for name, value in expected.items():
        if read(directory / name) != value:
            raise ValueError(
                "Input differs from the fixed source subset: " + name
            )
    if len(points) != profile["counts"]["scoring_points"]:
        raise ValueError("Incomplete scoring universe")
    return profile, binding, rows, points


def audit_record(run, point):
    """Check the saved instance and event files for run and point.

    Return None for valid records or a missing/invalid status
    dictionary.
    """
    import zipfile

    import numpy as np

    base = Path(run) / "queries" / point["video"] / point["query_id"]
    try:
        with np.load(
            base / "instances" / f"{point['frame']:06d}.npz",
            allow_pickle=False,
        ) as f:
            instances = f["masks"]
            if instances.shape[1:] != tuple(point["shape"]):
                raise ValueError("Invalid saved instance dimensions")
        event = read(base / "events" / f"{point['frame']:06d}.json")
        if event["max_source_tick_accessed"] > point["frame"]:
            raise ValueError("Future-frame access in saved event")
        event["total_seconds"]

    except FileNotFoundError as e:
        return dict(status="missing", error=str(e))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        EOFError,
        zipfile.BadZipFile,
    ) as e:
        return dict(status="invalid", error=str(e))
    return None


def run_identity(run, inputs):
    """Return phase path and metadata for run and inputs."""
    run = Path(run)
    phase = run / "predictions_run"
    config = read(run / "execution_config.json")
    binding = read(phase / "binding.json")
    if binding.get("GT_read") is not False or binding.get("method") not in [
        "sekai"
    ]:
        raise ValueError("Expected a GT-free structured Sekai run binding")
    if binding["manifest"] != read(Path(inputs) / "inference_manifest.json"):
        raise ValueError("Run was not bound to this exact inference manifest")
    if binding["config"] != config:
        raise ValueError("Run configuration changed")
    return phase, dict(
        mode="sekai_run_with_record_audit",
        method=binding["method"],
        run_complete_marker_present=(phase / "complete.json").is_file(),
    )


def score(inputs, data_root, output, run=None, prediction_root=None):
    """Score saved predictions against prepared inputs.

    Read GT from data_root, write reports to output, and return
    summary counts.
    """
    from check_examples import check
    from evaluate import aggregate, score_point, summarize

    profile, binding, inference, points = load_inputs(inputs)
    if Path(output).exists():
        raise FileExistsError("Use a new evaluation output directory")
    if (run is None) == (prediction_root is None):
        raise ValueError("Select exactly one prediction source")
    check(EXAMPLES, data_root)
    phase = None
    if run is not None:
        phase, identity = run_identity(run, inputs)
        prediction_root = phase / "predictions"
    else:
        identity = dict(
            mode="saved_union_png_only",
            profile="saved_union_pngs",
            note=("Input layout: video/query/frame.png union masks."),
        )
    rows, sidecar_failures = [], []
    for point, pred in zip(points, predictions_for(points)):
        issue = audit_record(phase, point) if phase is not None else None
        if issue:
            sidecar_failures.append(
                dict(query_id=point["query_id"], frame=point["frame"], **issue)
            )
            pred = None if issue["status"] == "missing" else dict(invalid=True)
        row = score_point(point, pred, data_root, prediction_root)
        if issue:
            row["error"] = issue["error"]
        rows.append(row)
    counts = collections.Counter(r["prediction_status"] for r in rows)
    groups = summarize(rows)
    for name, group in groups.items():
        if name == "all":
            rs = rows
        elif name == "history":
            rs = [r for r in rows if r.get("history") is True]
        elif name == "Other":
            rs = [
                r
                for r in rows
                if r["mode"] not in ["NOW", "EVER", "LATEST", "NEVER"]
            ]
        else:
            field, value = name.split(":", 1)
            rs = [r for r in rows if str(r.get(field)) == value]
        c = collections.Counter(r["prediction_status"] for r in rs)
        group.update(
            coverage=c["valid"] / len(rs) if rs else None,
            missing=c["missing"],
            invalid=c["invalid"],
        )
    by_video = {}
    for vid in profile["videos"]:
        rs = [r for r in rows if r["video"] == vid]
        c = collections.Counter(r["prediction_status"] for r in rs)
        by_video[vid] = dict(
            records=len(rs),
            coverage=c["valid"] / len(rs),
            missing=c["missing"],
            invalid=c["invalid"],
            **aggregate(rs),
        )
    result = dict(
        subset_id=profile["subset_id"],
        scope="example_subset_only",
        full_isekai_result=False,
        created=datetime.datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
        policy="original_record_failure_zero",
        units="0_to_1",
        prediction_source=identity,
        videos=len(profile["videos"]),
        queries=len(inference),
        records=len(rows),
        excluded_by_original_score_flag=binding[
            "excluded_by_original_score_flag"
        ],
        coverage=counts["valid"] / len(rows),
        missing=counts["missing"],
        invalid=counts["invalid"],
        groups=groups,
        per_video=by_video,
        sidecar_failures=sidecar_failures,
        per_point=rows,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "scores.json", result)
    summary = {
        k: v
        for k, v in result.items()
        if k not in ["per_point", "sidecar_failures"]
    }
    write(output / "summary.json", summary)
    lines = [
        "# Isekai seven-video example subset",
        "",
        "Evaluation scope: seven Isekai example videos.",
        "",
        (
            "Scoring: missing/invalid predictions receive zero. "
            f"Prediction source: `{identity['mode']}`."
        ),
        "",
        (
            f"7 videos / 32 queries / {len(rows)} scoring points. "
            f"Coverage: {result['coverage']:.2%}; "
            f"missing: {result['missing']}; invalid: {result['invalid']}."
        ),
        "",
        (
            "| Group | Points | Videos | J (%) | F (%) | J&F (%) | Coverage | "
            "Missing | Invalid |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, g in groups.items():

        def metric(key):
            """Return the percentage score for key, or n/a."""
            return "n/a" if g[key] is None else f"{100 * g[key]:.4f}"

        coverage = "n/a" if g["coverage"] is None else f"{g['coverage']:.2%}"
        lines.append(
            f"| {name} | {g['records']} | {g['videos']} | {metric('J')} | "
            f"{metric('F')} | {metric('JF')} | {coverage} | "
            f"{g['missing']} | {g['invalid']} |"
        )
    lines += [
        "",
        (
            "JSON metrics use 0–1 units; this table uses percent. Each group "
            "first filters points, then averages within video and equally "
            "across its represented videos."
        ),
        "",
        (
            "History is the recorded NOW/EVER/NEVER subset. Evidence-age "
            "timing proxies and NEVER anchors are inherited from the supplied "
            "classification. Evidence age applies to nonempty GT."
        ),
        "",
        (
            "Per-video and per-point scores, errors and prediction coverage "
            "are in scores.json. Missing records remain in the denominator."
        ),
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return {
        k: v for k, v in summary.items() if k not in ["groups", "per_video"]
    }


def main():
    """Prepare or score CLI inputs and print a JSON summary."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser(
        "prepare",
        help=("Validate assets and prepare inference and scoring inputs"),
    )
    prep.add_argument("--assets", required=True)
    prep.add_argument("--output", required=True)
    ev = sub.add_parser(
        "score",
        help="Compute CPU metrics for every saved-prediction scoring point",
    )
    ev.add_argument("--inputs", required=True)
    ev.add_argument("--data-root", required=True)
    source = ev.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run",
        help=(
            "tools/infer.py output; checks manifest/config identity and "
            "instance/event sidecars"
        ),
    )
    source.add_argument(
        "--prediction-root",
        help="Explicit PNG-only mode: video/query/000000.png",
    )
    ev.add_argument("--output", required=True)
    a = p.parse_args()
    if a.command == "prepare":
        result = prepare(a.assets, a.output)
    else:
        result = score(
            a.inputs, a.data_root, a.output, a.run, a.prediction_root
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
