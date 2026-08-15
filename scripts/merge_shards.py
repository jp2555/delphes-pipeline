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

Parallel by OUTPUT FILE. The work is dominated by codec — every shard is decompressed and
recompressed with zstd, which runs at a few hundred MB/s on one core — so the groups are
planned up front from the shards' own sizes and each output file is written by its own
process. That needs the boundaries decided BEFORE writing, which is why the grouping uses
input sizes rather than watching the output grow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import sys
from concurrent.futures import ProcessPoolExecutor

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_shards  # noqa: E402


def _plan(outdir):
    with open(os.path.join(outdir, "_plan", "manifest.json")) as fh:
        return json.load(fh)["shards"]


# kl-1p00, kl-0p00, kl-m2p50 ... the generated kappa_lambda is encoded ONLY in the
# input path. The ntuplizer does not carry it (schema.SCALARS has no kl field), so a
# merged signal sample with every kl point in it would be unusable for NSBI -- the
# method is entirely about ratios BETWEEN kl hypotheses. Recover it here, from the
# plan, and write it as a column so the merged file is self-describing.
_KL_RE = re.compile(r"kl-(m?)(\d+)p(\d+)")


def _kl_of(entry):
    """The kappa_lambda a shard was generated at, or None for samples without one.

    Raises if one shard spans two kl points: the events would then be unlabelable,
    and silently picking one would poison the training targets.
    """
    vals = set()
    for f in entry.get("files", ()):
        m = _KL_RE.search(f)
        if m:
            vals.add((-1.0 if m.group(1) else 1.0) * float(f"{m.group(2)}.{m.group(3)}"))
    if len(vals) > 1:
        raise ValueError(f"shard {entry['sample']}.{entry['shard']:04d} spans multiple "
                         f"kl points {sorted(vals)} — cannot label its events")
    return vals.pop() if vals else None


def _group(entries, target_bytes):
    """Partition shards into output groups by their INPUT size.

    Boundaries have to be fixed before any writing starts, otherwise the groups cannot be
    handed to separate processes. Input size is a good proxy: the output carries the same
    data through the same codec.
    """
    groups, cur, acc = [], [], 0
    for e in entries:
        cur.append(e)
        acc += os.path.getsize(e["out"])
        if acc >= target_bytes:
            groups.append(cur); cur, acc = [], 0
    if cur:
        groups.append(cur)
    return groups


def write_group(job):
    """Write ONE output file from a group of shards. Module level, so it can be pickled
    to a worker process."""
    name, part, entries, dest = job
    out_path = os.path.join(dest, f"{name}.{part:04d}.parquet")
    writer, schema, rows = None, None, 0
    try:
        for e in entries:
            kl = _kl_of(e)
            if kl is None:
                t = pq.read_table(e["out"])          # fast path: straight arrow
            else:
                # Appending through raw arrow leaves the file's awkward metadata not
                # mentioning kl, and ak.from_parquet then refuses to read it at all.
                # Adding the field through awkward keeps the metadata consistent. Only
                # signal pays this; ttbar (the 145 GB one) stays on the fast path.
                a = ak.from_parquet(e["out"])
                a = ak.with_field(a, np.full(len(a), kl, dtype="float32"), "kl")
                t = ak.to_arrow_table(a)
            if schema is None:
                schema = t.schema
                writer = pq.ParquetWriter(out_path, schema, compression="zstd")
            elif not t.schema.equals(schema):
                # coercing here would silently drop or reorder columns; a schema change
                # means the shards were made by different code and must not be mixed
                raise ValueError(
                    f"{name}: schema mismatch at {e['out']}\n"
                    f"        expected {schema.names}\n        got      {t.schema.names}")
            writer.write_table(t)
            rows += t.num_rows
    finally:
        if writer is not None:
            writer.close()
    return out_path, rows


def merge_sample(name, entries, dest, target_bytes, jobs=1):
    """Stream ``entries`` into ``dest/name.NNNN.parquet`` files of ~target size."""
    groups = _group(entries, target_bytes)
    todo = [(name, i, g, dest) for i, g in enumerate(groups)]
    if jobs <= 1 or len(todo) <= 1:
        results = [write_group(j) for j in todo]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(todo))) as ex:
            results = list(ex.map(write_group, todo))
    return [r[0] for r in results], sum(r[1] for r in results)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="the campaign directory holding the shards")
    ap.add_argument("--dest", default=None, help="where to write (default <out>/merged)")
    ap.add_argument("--target-gb", type=float, default=5.0)
    ap.add_argument("--jobs", type=int, default=0,
                    help="processes; one per output file. 0 = min(cpu_count, files). The "
                         "work is codec-bound (zstd decompress+recompress of ~82 GB), so "
                         "this scales close to linearly until the filesystem saturates.")
    ap.add_argument("--sample", action="append", metavar="NAME",
                    help="merge only this sample (repeatable). Lets a finished sample "
                         "be merged for real while another is still being recovered.")
    ap.add_argument("--force", action="store_true",
                    help="merge even when shards are missing or duplicated — the merged "
                         "sample will then be silently short")
    args = ap.parse_args(argv)

    outdir = os.path.abspath(args.out)
    dest = os.path.abspath(args.dest or os.path.join(outdir, "merged"))
    os.makedirs(dest, exist_ok=True)

    plan = _plan(outdir)
    by_sample = {}
    for e in plan:
        by_sample.setdefault(e["sample"], []).append(e)

    if args.sample:
        unknown = set(args.sample) - set(by_sample)
        if unknown:
            raise SystemExit(f"[merge] no such sample: {', '.join(sorted(unknown))}")
        by_sample = {k: v for k, v in by_sample.items() if k in args.sample}

    # Completeness is judged over the samples actually being merged: a gap in ttbar is
    # no reason to block a finished signal merge, but it must still block ttbar's. A
    # duplicated shard id is just as disqualifying as a missing one -- it double-counts.
    report = {}
    t0 = time.perf_counter()
    make_shards.verify(outdir, report=report,
                       samples=list(by_sample) if args.sample else None)
    print(f"[merge] audit took {time.perf_counter() - t0:.0f}s")
    bad = {s for s, _ in report["missing"]} | {s for s, _ in report["dup"]}
    blocked = sorted(bad & set(by_sample))
    if blocked and not args.force:
        print(f"[merge] refusing to merge an incomplete sample: {', '.join(blocked)} "
              f"(--force to override, or --sample to merge only the finished ones)")
        return 1

    summary = {}
    for name, entries in by_sample.items():
        entries = [e for e in sorted(entries, key=lambda x: x["shard"])
                   if os.path.exists(e["out"])]
        jobs = args.jobs or (os.cpu_count() or 1)
        t1 = time.perf_counter()
        try:
            files, rows = merge_sample(name, entries, dest, args.target_gb * 1e9, jobs=jobs)
        except ValueError as exc:
            # raised in a worker; surface it as a clean abort rather than a traceback
            raise SystemExit(f"[merge] {exc}")
        size = sum(os.path.getsize(f) for f in files)
        dt = max(time.perf_counter() - t1, 1e-9)
        print(f"[merge] {name}: {len(entries)} shards -> {len(files)} files, "
              f"{rows:,} events, {size/1e9:.1f} GB in {dt:.0f}s "
              f"({size/1e6/dt:.0f} MB/s out, {min(jobs, len(files))} workers)")
        summary[name] = {"shards": len(entries), "files": files, "events": rows,
                         "bytes": size,
                         "maps": sorted({e["maps"] for e in entries}),
                         "subtree": sorted({e.get("subtree") for e in entries})}

    mpath = os.path.join(dest, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as fh:
            prior = json.load(fh)
        prior.update(summary)
        summary = prior
    with open(mpath, "w") as fh:
        json.dump(summary, fh, indent=2)
    total = sum(v["events"] for v in summary.values())
    print(f"[merge] {total:,} events total -> {dest}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
