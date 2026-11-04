"""Reconstruct example assets from local Isekai or GroundMoRe files."""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def relative_path(root, name):
    """Resolve name within root and validate the resulting path."""
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Asset paths must stay within the selected root")
    return path


def from_groundmore(source, output):
    """Write source RGB frames and binary instance masks to output."""
    import numpy as np
    from PIL import Image

    examples = ROOT / "examples"
    index = json.loads((examples / "index.json").read_text(encoding="utf8"))
    for spec in index:
        data = json.loads(
            (examples / (spec["video_id"] + ".json")).read_text(
                encoding="utf8"
            )
        )
        dataset, clip = data["video"]["source"].split("/", 1)
        if dataset != "groundmore":
            raise ValueError("Expected a GroundMoRe source clip")
        root = relative_path(source, clip)
        for frame in data["video"]["frames"]:
            name = frame["source_frame_id"]
            image_path = relative_path(root, "images/" + name + ".jpg")
            mask_path = relative_path(root, "masks/" + name + ".png")
            with Image.open(image_path) as image:
                if image.size != (frame["width"], frame["height"]):
                    raise ValueError("Source image dimensions differ")
            with Image.open(mask_path) as image:
                labels = np.asarray(image)
            if labels.shape != (frame["height"], frame["width"]):
                raise ValueError("Source mask dimensions differ")
            target = relative_path(output, frame["image"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_path, target)
            for obj in frame["objects"]:
                mask = labels == int(obj["instance_id"])
                yy, xx = np.nonzero(mask)
                box = (
                    [
                        int(xx.min()),
                        int(yy.min()),
                        int(xx.max()) + 1,
                        int(yy.max()) + 1,
                    ]
                    if len(xx)
                    else None
                )
                if int(mask.sum()) != obj["pixels"] or box != obj["bbox_xyxy"]:
                    raise ValueError("Source instance geometry differs")
                target = relative_path(output, obj["mask"])
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask.astype("uint8") * 255).save(target)


def main():
    """Write example assets from the selected local source format."""
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", help="Isekai asset directory")
    source.add_argument(
        "--groundmore-root", help="GroundMoRe annotations directory"
    )
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    if a.groundmore_root:
        from_groundmore(a.groundmore_root, out)
    else:
        path = ROOT / "examples/asset_map.json"
        for x in json.loads(path.read_text(encoding="utf8")):
            src = relative_path(a.source_root, x["path"])
            dest = relative_path(out, x["path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
    print("Example RGB frames and instance masks written to " + str(out))


if __name__ == "__main__":
    main()
