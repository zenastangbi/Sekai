"""Writer command-line execution and memory storage."""

import argparse
from pathlib import Path

import memory as contract
from common import bind, chunks, dump, read, validate_rows


def main():
    """Write per-chunk memory records from CLI inputs."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cfg = read(a.config)
    rows = read(a.manifest)
    validate_rows(rows)
    out = Path(a.output)
    identity = dict(config=cfg, manifest=rows, GT_read=False)
    bind(out / "binding.json", identity)
    from base_model import load
    from generation import generate
    from writer import memory_batch

    route = None
    for row in rows:
        previous = ""
        rendered = ""
        cutoff = -1
        failed = []
        for c in chunks(row):
            dest = (
                out
                / "queries"
                / row["source_video"]
                / row["expression_id"]
                / "writer"
                / f"{c['call']:04d}.json"
            )
            if dest.exists():
                z = read(dest)
                assert z["call"] == c and z["writer_memory_in"] == previous
            else:
                if route is None:
                    route = load(cfg["writer"])
                attempts = []
                valid = False
                parsed = None
                for retry in range(3):
                    batch, _, audit = memory_batch(route, row, c, previous)
                    assert (
                        not audit["GT_in_input"]
                        and audit["real_indices"] == c["real_indices"]
                    )
                    g = generate(route, batch, cfg["writer"]["seed"], 640)
                    del batch
                    try:
                        parsed = contract.parse(
                            g["raw"],
                            tokenizer=route.processor.tokenizer,
                            observed_indices=list(
                                range(c["target_index"] + 1)
                            ),
                        )
                        valid = True
                        error = None
                    except (ValueError, TypeError) as e:
                        error = repr(e)
                    attempts.append(
                        dict(
                            generation=g,
                            input=audit,
                            retry=retry,
                            validation_error=error,
                        )
                    )
                    if valid:
                        break
                if valid:
                    canonical = parsed["canonical"]
                    newtext = parsed["rendered"]
                    newobj = parsed["structured"]
                    newcutoff = c["target_index"]
                else:
                    fallback = contract.failed_update(previous)
                    assert not fallback["run_locator"]
                    canonical = fallback["writer_memory_out"]
                    newtext = fallback["memory_out"]
                    newobj = fallback["structured"]
                    newcutoff = cutoff
                    failed = failed + [c["real_indices"]]
                z = dict(
                    row_id=row["id"],
                    call=c,
                    writer_memory_in=previous,
                    writer_memory_out=canonical,
                    writer_update_valid=valid,
                    locator_allowed=valid,
                    memory_in=rendered,
                    memory_out=newtext,
                    structured=newobj,
                    valid=valid,
                    attempts=attempts,
                    raw_reply=g["raw"],
                    memory_cutoff=newcutoff,
                    failed_intervals=failed,
                    GT_in_input=False,
                    body_tokens=parsed["body_tokens"] if valid else None,
                    rendered_tokens=parsed["rendered_tokens"]
                    if valid
                    else None,
                )
                dump(dest, z)
            previous = z["writer_memory_out"]
            rendered = z["memory_out"]
            cutoff = z["memory_cutoff"]
            failed = z["failed_intervals"]
            dump(
                out / "progress.json",
                dict(
                    phase="structured_writer",
                    query=row["id"],
                    call=c["call"],
                    valid=z["valid"],
                ),
            )
    dump(out / "writer_complete.json", dict(queries=len(rows), **identity))


if __name__ == "__main__":
    main()
