"""Command-line orchestration of inference stages."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
ROOT = Path(__file__).resolve().parents[1]


def resolve_config(path):
    """Return validated asset paths from config at path."""
    from common import read

    cfg = read(path)

    def resolve(value):
        """Expand environment-variable path placeholders in value."""
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if (
            isinstance(value, str)
            and value.startswith("${")
            and value.endswith("}")
        ):
            name = value[2:-1]
            if not os.environ.get(name):
                raise ValueError("Set the required asset path: " + name)
            return str(Path(os.environ[name]).expanduser().resolve())
        return value

    cfg = resolve(cfg)
    for component in ["writer", "locator"]:
        model = Path(cfg[component]["model"])
        if not (model / "config.json").is_file():
            raise FileNotFoundError(
                "A local Qwen3-VL-8B-Instruct model directory is required"
            )
    for key in ["checkpoint", "sam_checkpoint"]:
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(
                "Required asset path must identify a file: " + key + ""
            )
    if not (Path(cfg["sam_source"]) / "sam3/model_builder.py").is_file():
        raise FileNotFoundError(
            "The compatible causal SAM source tree is required"
        )
    return cfg


def main():
    """Write inference-stage outputs from CLI inputs."""
    from common import bind, chunks, read, validate_rows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "configs/inference.json")
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = resolve_config(args.config)
    rows = read(args.manifest)
    validate_rows(rows)
    for row in rows:
        for frame in row["frames"]:
            if not Path(frame).is_file():
                raise FileNotFoundError("An input RGB frame is missing")
    out = Path(args.output).resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError("Store runtime outputs outside the source package")
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "input_config.json"
    bind(config_path, cfg)
    env = os.environ.copy()
    env["SEKAI_SAM_CHECKPOINT"] = cfg["sam_checkpoint"]
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "core"), cfg["sam_source"]]
    )

    def run(file, *arguments):
        """Execute file with arguments in the configured environment."""
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "core" / file),
                *map(str, arguments),
            ],
            env=env,
            check=True,
        )

    cache = out / "writer_cache"
    run(
        "structured_writer.py",
        "--config",
        config_path,
        "--manifest",
        args.manifest,
        "--output",
        cache,
    )
    if not (cache / "writer_complete.json").is_file():
        raise RuntimeError("Writer has not completed")
    if read(cache / "binding.json") != dict(
        config=cfg, manifest=rows, GT_read=False
    ):
        raise ValueError("Writer cache belongs to different inputs")
    for row in rows:
        for call in chunks(row):
            path = (
                cache
                / "queries"
                / row["source_video"]
                / row["expression_id"]
                / "writer"
                / f"{call['call']:04d}.json"
            )
            record = read(path)
            if record["call"] != call or record["GT_in_input"]:
                raise ValueError("Invalid writer cache entry")
    cfg["writer_cache"] = str(cache)
    execution = out / "execution_config.json"
    bind(execution, cfg)
    for phase in ["locator", "sam"]:
        run(
            "inference.py",
            "--config",
            execution,
            "--manifest",
            args.manifest,
            "--output",
            out / "predictions_run",
            "--phase",
            phase,
        )


if __name__ == "__main__":
    main()
