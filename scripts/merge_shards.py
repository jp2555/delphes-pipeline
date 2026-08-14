"""Consolidate the per-shard parquet into a small, complete, auditable dataset.

    pixi run python scripts/merge_shards.py --out /ceph/jpan/ntuples --target-gb 5

1394 shards is an awkward training input: a reader has to open every file, and nothing
about a directory listing says whether the set is COMPLETE. So this streams the shards
of each sample into a handful of larger files and records what went in.

It REFUSES to merge an incomplete or duplicated set unless forced. That check is the
whole reason the ntuplizer stamps a ``shard`` column: a campaign missing 3 of 1244 tt̄
shards looks exactly like a finished one on disk, and the resulting sample is quietly
0.2% short — which is invisible in a shape comparison and wrong in a normalisation.

Streaming, not loading: the campaign is ~82 GB, so shards are appended row-group by
row-group through a ParquetWriter and only one shard is resident at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_shards  # noqa: E402


def _plan(outdir):
    with open(os.path.join(outdir, "_plan", "manifest.json")) as fh:
        return json.load(fh)["shards"]


def merge_sample(name, entries, dest, target_bytes):
    """Stream ``entries`` into ``dest/name.NNNN.parquet`` files of ~target size."""
    written, part, rows, writer, schema = [], 0, 0, None, None
    out_path = None
    try:
        for e in entries:
            t = pq.read_table(e["out"])
            if schema is None:
                schema = t.schema
            elif not t.schema.equals(schema):
                # coercing here would silently drop or reorder columns; a schema change
                # means the shards were made by different code and must not be mixed
                raise SystemExit(
                    f"[merge] {name}: schema mismatch at {e['out']}\n"
                    f"        expected {schema.names}\n        got      {t.schema.names}")
            if writer is None:
                out_path = os.path.join(dest, f"{name}.{part:04d}.parquet")
                writer = pq.ParquetWriter(out_path, schema, compression="zstd")
            writer.write_table(t)
            rows += t.num_rows
            if os.path.getsize(out_path) >= target_bytes:
                writer.close(); writer = None
                written.append(out_path); part += 1
        if writer is not None:
            writer.close()
            written.append(out_path)
    finally:
        if writer is not None:
            writer.close()
    return written, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="the campaign directory holding the shards")
    ap.add_argument("--dest", default=None, help="where to write (default <out>/merged)")
    ap.add_argument("--target-gb", type=float, default=5.0)
    ap.add_argument("--force", action="store_true",
                    help="merge even when shards are missing or duplicated — the merged "
                         "sample will then be silently short")
    args = ap.parse_args(argv)

    outdir = os.path.abspath(args.out)
    dest = os.path.abspath(args.dest or os.path.join(outdir, "merged"))
    os.makedirs(dest, exist_ok=True)

    if make_shards.verify(outdir) != 0 and not args.force:
        print("[merge] refusing to merge an incomplete set (--force to override)")
        return 1

    plan = _plan(outdir)
    by_sample = {}
    for e in plan:
        by_sample.setdefault(e["sample"], []).append(e)

    summary = {}
    for name, entries in by_sample.items():
        entries = [e for e in sorted(entries, key=lambda x: x["shard"])
                   if os.path.exists(e["out"])]
        files, rows = merge_sample(name, entries, dest, args.target_gb * 1e9)
        size = sum(os.path.getsize(f) for f in files)
        print(f"[merge] {name}: {len(entries)} shards -> {len(files)} files, "
              f"{rows:,} events, {size/1e9:.1f} GB")
        summary[name] = {"shards": len(entries), "files": files, "events": rows,
                         "bytes": size,
                         "maps": sorted({e["maps"] for e in entries}),
                         "subtree": sorted({e.get("subtree") for e in entries})}

    with open(os.path.join(dest, "manifest.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    total = sum(v["events"] for v in summary.values())
    print(f"[merge] {total:,} events total -> {dest}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
