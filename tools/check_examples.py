"""Example-annotation and asset validation."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(directory, assets=None):
    """Validate annotations in directory and optional assets.

    Return label count and asset status, or raise for inconsistent
    records.
    """
    index = json.loads((directory / "index.json").read_text())
    count = 0
    missing = []
    frame_maps = {}
    for spec in index:
        d = json.loads(
            (directory / (spec["video_id"] + ".json")).read_text(
                encoding="utf8"
            )
        )
        v = d["video"]
        frames = v["frames"]
        assert [f["published_index"] for f in frames] == list(
            range(len(frames))
        )
        frame_maps[v["video_id"]] = {f["source_frame_id"]: f for f in frames}
        queries = {q["query_id"] for q in d["queries"]}
        events = {e["event_id"]: e for e in d["events"]["events"]}
        known = {o["instance_id"] for f in frames for o in f["objects"]}
        assert all(
            str(p["instance_id"]) in known
            for e in events.values()
            for p in e["participants"]
        )
        for r in d["labels"]:
            assert r["query_id"] in queries and r["video_id"] == v["video_id"]
            assert set(r["output_ids"]) == set(r["eligible_ids"]) & set(
                r["visible_ids"]
            )
            fr = frames[r["published_index"]]
            objects = {o["instance_id"]: o for o in fr["objects"]}
            assert set(r["output_ids"]) <= objects.keys()
            assert {g["instance_id"] for g in r["gt"]} == set(r["output_ids"])
            for g in r["gt"]:
                assert objects[g["instance_id"]]["mask"] == g["mask"]
            for e in r.get("supported_event_ids", []):
                assert e in events
            count += 1
        for q in queries:
            assert {
                r["published_index"] for r in d["labels"] if r["query_id"] == q
            } == set(range(len(frames)))
    if assets:
        for a in json.loads((directory / "asset_map.json").read_text()):
            p = Path(assets) / a["path"]
            if not p.is_file():
                missing.append(a["path"])
                continue
            from PIL import Image

            with Image.open(p) as image:
                image.load()
                frame = frame_maps[a["video_id"]][a["source_frame_id"]]
                if image.size != (frame["width"], frame["height"]):
                    raise ValueError(
                        "Asset dimensions do not match annotation: "
                        + a["path"]
                    )
                if a["kind"] == "mask" and len(image.getbands()) != 1:
                    raise ValueError(
                        "Expected a single-channel mask: " + a["path"]
                    )
        if missing:
            raise FileNotFoundError(f"{len(missing)} required assets missing")
    return dict(status="PASS", label_rows=count, assets_present=bool(assets))


def main():
    """Validate CLI inputs and print annotation and asset status."""
    p = argparse.ArgumentParser()
    p.add_argument("--examples", default=str(ROOT / "examples"))
    p.add_argument("--assets")
    a = p.parse_args()
    print(json.dumps(check(Path(a.examples), a.assets)))


if __name__ == "__main__":
    main()
