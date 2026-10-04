"""JSON files, inference manifests, and chunk schedules."""

import json
import re
from pathlib import Path


def read(path):
    """Return the JSON value stored in the UTF-8 file at path."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    """Serialize value as UTF-8 JSON and atomically replace path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def component(value):
    """Return value as a validated path component."""
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Unsafe path component: {value!r}")
    return value


def chunks(row):
    """Return chunk records from row frame paths and source indices.

    Each record contains its call index, frame range, and endpoint.
    """
    n = len(row["frames"])
    if n < 1:
        raise ValueError("Empty video")
    spans = [(0, 1)] + [(a, min(a + 8, n)) for a in range(1, n, 8)]
    return [
        dict(
            call=i,
            real_indices=list(range(a, b)),
            paths=row["frames"][a:b],
            source_indices=row["source_indices"][a:b],
            target_index=b - 1,
            cutoff_in=a - 1,
        )
        for i, (a, b) in enumerate(spans)
    ]


def expected_paths(rows):
    """Return video/query/frame PNG paths from manifest rows."""
    result = set()
    for row in rows:
        for frame in row["frame_names"]:
            path = (
                "/".join(
                    component(x)
                    for x in (row["source_video"], row["expression_id"], frame)
                )
                + ".png"
            )
            if path in result:
                raise ValueError("Duplicate expression/frame output")
            result.add(path)
    return result


def validate_rows(rows):
    """Validate manifest rows; return None or raise ValueError."""
    allowed = {
        "id",
        "source_video",
        "expression_id",
        "raw_query",
        "frame_names",
        "frames",
        "source_indices",
    }
    for row in rows:
        if set(row) != allowed:
            raise ValueError("Manifest fields must match the inference schema")
        n = len(row["frames"])
        if len(row["frame_names"]) != n or row["source_indices"] != list(
            range(n)
        ):
            raise ValueError("Frame identity mismatch")
        chunks(row)
    expected_paths(rows)


def bind(path, value):
    """Write value to path or compare it with the stored value."""
    if Path(path).exists():
        if read(path) != value:
            raise ValueError("Frozen run identity changed")
    else:
        dump(path, value)
