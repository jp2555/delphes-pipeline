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
#: dataset_id for events whose shard straddled two primary datasets. NOT a dataset:
#: their cross section is genuinely ambiguous, so they are marked and counted rather
#: than attributed to whichever contributed more files.
MIXED_DATASET = -1

_KL_RE = re.compile(r"kl-(m?)(\d+)p(\d+)")


#: the CMS primary dataset a shard's inputs came from. One `--sample` can span several
#: (ttbar globs TTto2L2Nu + TTto4Q + TTtoLNu2Q; DY is jet- and mass-binned), and those
#: carry DIFFERENT cross sections -- 98.0 / 419.7 / 405.7 pb for the three ttbar channels.
#: Merging them without a per-event label makes the sample unnormalisable, exactly as
#: globbing the kl directories did. Recovered here, from the plan, for the same reason.
_DATASET_RE = re.compile(r"/([A-Za-z0-9][^/]*?_TuneCP5_[^/]*?)_Delphes")


def _dataset_of(entry):
    """The primary dataset name(s) a shard's input files belong to."""
    out = set()
    for f in entry.get("files", ()):
        m = _DATASET_RE.search(f)
        if m:
            out.add(m.group(1))
    return sorted(out)


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
    name, part, entries, dest, datasets, allow_mixed = job
    out_path = os.path.join(dest, f"{name}.{part:04d}.parquet")
    writer, schema, rows = None, None, 0
    sumw = sumw_sf = 0.0
    mixed = 0
    try:
        for e in entries:
            kl = _kl_of(e)
            ds = _dataset_of(e)
            if len(ds) > 1 and not allow_mixed:
                raise ValueError(
                    f"shard {e['sample']}.{e['shard']:04d} straddles {len(ds)} datasets "
                    f"({', '.join(ds)}) — they have different cross sections and the "
                    f"events cannot be labelled. Shards are cut on accumulated BYTES "
                    f"across the whole file list, so one straddles each dataset "
                    f"boundary. Pass --allow-mixed-datasets to label those events -1 "
                    f"(unnormalisable, and counted in the manifest) instead of aborting.")
            if len(ds) > 1:
                dsid = MIXED_DATASET      # sentinel: known-unattributable, never guessed
            else:
                dsid = datasets.setdefault(ds[0], len(datasets)) if ds else None
            if kl is None and dsid is None:
                t = pq.read_table(e["out"])          # fast path: straight arrow
            else:
                # Appending through raw arrow leaves the file's awkward metadata not
                # mentioning kl, and ak.from_parquet then refuses to read it at all.
                # Adding the field through awkward keeps the metadata consistent. Only
                # signal pays this; ttbar (the 145 GB one) stays on the fast path.
                a = ak.from_parquet(e["out"])
                if kl is not None:
                    a = ak.with_field(a, np.full(len(a), kl, dtype="float32"), "kl")
                if dsid is not None:
                    a = ak.with_field(a, np.full(len(a), dsid, dtype="int16"), "dataset_id")
                    if dsid == MIXED_DATASET:
                        mixed += len(a)
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
            if "genWeight" in t.column_names:
                # sum WITH sign: NLO weights are negative for a real fraction of events,
                # and sigma_eff's numerator and denominator must both include them
                sumw += float(np.asarray(t.column("genWeight")).sum())
            if "lepton_sf" in t.column_names:
                sumw_sf += float((np.asarray(t.column("genWeight"))
                                  * np.asarray(t.column("lepton_sf"))).sum())
    finally:
        if writer is not None:
            writer.close()
    return out_path, rows, sumw, sumw_sf, mixed


def merge_sample(name, entries, dest, target_bytes, jobs=1, datasets=None,
                 allow_mixed=False):
    """Stream ``entries`` into ``dest/name.NNNN.parquet`` files of ~target size."""
    datasets = {} if datasets is None else datasets
    # assign ids up front so every worker agrees on them
    for e in entries:
        for d in _dataset_of(e):
            datasets.setdefault(d, len(datasets))
    groups = _group(entries, target_bytes)
    todo = [(name, i, g, dest, dict(datasets), allow_mixed)
            for i, g in enumerate(groups)]
    if jobs <= 1 or len(todo) <= 1:
        results = [write_group(j) for j in todo]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(todo))) as ex:
            results = list(ex.map(write_group, todo))
    return ([r[0] for r in results], sum(r[1] for r in results),
            sum(r[2] for r in results), sum(r[3] for r in results),
            sum(r[4] for r in results))


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
    ap.add_argument("--allow-mixed-datasets", action="store_true",
                    help="label events from a shard spanning two primary datasets with "
                         "dataset_id=-1 instead of aborting. Those events cannot be "
                         "normalised (the cross sections differ) and must be dropped "
                         "downstream; the count is recorded in the manifest.")
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
            datasets = {}
            files, rows, sumw, sumw_sf, mixed = merge_sample(
                name, entries, dest, args.target_gb * 1e9, jobs=jobs, datasets=datasets,
                allow_mixed=args.allow_mixed_datasets)
        except ValueError as exc:
            # raised in a worker; surface it as a clean abort rather than a traceback
            raise SystemExit(f"[merge] {exc}")
        size = sum(os.path.getsize(f) for f in files)
        dt = max(time.perf_counter() - t1, 1e-9)
        print(f"[merge] {name}: {len(entries)} shards -> {len(files)} files, "
              f"{rows:,} events, {size/1e9:.1f} GB in {dt:.0f}s "
              f"({size/1e6/dt:.0f} MB/s out, {min(jobs, len(files))} workers)")
        planned = sum(1 for e in plan if e["sample"] == name)
        # sigma_eff = sigma_gen * (sum w on PROCESSED shards / sum w generated over the
        # SAME shard set). Recording both the sum and the shard accounting is what lets
        # that be computed without ever quoting a generated count the sample does not
        # correspond to (the 4 lost ttbar shards are exactly this trap).
        summary[name] = {"shards": len(entries), "shards_planned": planned,
                         "files": files, "events": rows,
                         "sum_genweight": sumw,
                         # dataset_id column -> CMS primary dataset. Each carries its own
                         # cross section, so this is what makes the sample normalisable.
                         "datasets": {v: k for k, v in datasets.items()},
                         "mixed_dataset_events": mixed,
                         "sum_genweight_x_lepton_sf": sumw_sf,
                         "bytes": size,
                         "maps": sorted({e["maps"] for e in entries}),
                         "maps_sha": sorted({e.get("maps_sha") for e in entries}),
                         "subtree": sorted({e.get("subtree") for e in entries})}

    # Provenance uniformity is what the downstream unbinned CI depends on. Samples
    # corrected with DIFFERENT maps are different forward models, and a signal/background
    # ratio trained across them inherits the map difference as learned S/B shape. That may
    # be intended; it must never be silent.
    shas = {n: (v.get("maps_sha") or [None])[0] for n, v in summary.items()}
    if len(set(shas.values())) > 1:
        print("[merge] NOTE: samples were corrected with DIFFERENT map sets — "
              + ", ".join(f"{n}={v}" for n, v in sorted(shas.items())))
        print("[merge]   valid within each sample; a ratio trained ACROSS them carries "
              "the map difference. See docs/tuning_for_nsbi_audit.md")

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
