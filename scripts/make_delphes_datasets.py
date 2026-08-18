"""Emit `datasets.json`-format metadata for the merged Delphes ntuples.

    pixi run python scripts/make_delphes_datasets.py \
        --merged /ceph/jpan/ntuples_untuned/merged \
        --cms-datasets datasets.json \
        --out /ceph/jpan/ntuples_untuned/merged/datasets_delphes.json

Same schema the CMS side uses (``xsec``, ``nevents``, ``generator_weight``, ``nfiles``,
``sample_type``, ``era``, ``dbs``, ``nick``), so the NSBI config can read it unchanged --
plus the Delphes provenance the CMS file has no field for.

**Which number comes from where, and why it matters.**

``xsec`` is CMS's: the Delphes samples are simulated from the SAME primary datasets, so
the cross section is a property of the generator configuration and transfers exactly.
Matched by primary-dataset name via the ``dataset_id`` column the merge writes.

``nevents`` and ``generator_weight`` are OURS, computed over the shards we actually
PROCESSED. This is the sigma_eff rule: the denominator must correspond to the sample in
hand, never to a generated-event count the sample does not match. The tt-bar campaign
lost 7 of 1244 shards to a storage fault; quoting CMS's 470M generated events against
our processed subset would bias the yield by exactly the missing fraction. Both
``shards`` and ``shards_planned`` are carried so the shortfall stays visible.

``generator_weight`` follows CMS's convention -- the MEAN generator weight, i.e.
sum(genWeight)/nevents over the processed set, signed, so negative NLO weights are in
both the sum and the count.

Per-event normalisation is then the usual
``w_event * xsec * lumi / (nevents * generator_weight)``, evaluated per ``dataset_id``:
one ``--sample`` can span several primary datasets with different cross sections (ttbar
globs three decay channels; DY is jet- and mass-binned), which is what the ``dataset_id``
column exists to keep separable.

NB the signal cross section IS kappa_lambda dependent (5.535 / 2.493 / 7.280 fb at
kl = 0 / 1 / 5, i.e. sigma_SM = 34.1 fb at 13.6 TeV). Do NOT use
``convert_powheg_to_sbi.sigma_total()``: its coefficients (62.5, -44.3, 12.85) give
sigma_SM = 31.05 fb, the **13 TeV** value, and reintroduce exactly the sqrt(s) bridge the
public-anchor plan forbids.
"""

from __future__ import annotations

import argparse
import json
import os
import re


def _load_cms(path):
    with open(path) as fh:
        return json.load(fh)


def _match(cms: dict, dataset: str, prefer: str | None) -> tuple[str | None, dict]:
    """Find the CMS entry whose nick starts with this primary dataset name.

    Several processing campaigns share one primary dataset (a base sample, its _ext1
    statistics extension, a privately produced variant). They agree on ``xsec`` -- it is
    a generator property -- so any is fine for the cross section, but we prefer the one
    the samples were actually produced from and record which was used.
    """
    hits = [k for k in cms if k.startswith(dataset)]
    if not hits:
        return None, {}
    if prefer:
        pref = [k for k in hits if prefer in k]
        hits = pref or hits
    # deterministic, and prefer an entry that actually carries a cross section
    hits.sort(key=lambda k: (cms[k].get("xsec") is None, k))
    return hits[0], cms[hits[0]]


def scan_per_dataset(files) -> dict[int, tuple[int, float]]:
    """{dataset_id: (nevents, sum_genweight)} read from the merged files.

    Only the ``dataset_id`` and ``genWeight`` columns are read -- 6 bytes an event, so a
    300M-event sample costs a couple of GB of I/O rather than a full re-read. This is
    what makes a multi-dataset sample normalisable: the manifest holds only the sample
    TOTAL, and dividing that between three ttbar channels with cross sections spanning
    98-420 pb would be a fabricated number.
    """
    import numpy as np
    import pyarrow.parquet as pq

    acc: dict[int, list] = {}
    for f in files:
        pf = pq.ParquetFile(f)
        if "dataset_id" not in pf.schema_arrow.names:
            return {}
        cols = ["dataset_id"] + (["genWeight"] if "genWeight" in pf.schema_arrow.names
                                 else [])
        t = pq.read_table(f, columns=cols)
        ids = np.asarray(t.column("dataset_id"))
        w = (np.asarray(t.column("genWeight"), dtype=np.float64)
             if "genWeight" in cols else np.ones(len(ids)))
        for d in np.unique(ids):
            sel = ids == d
            a = acc.setdefault(int(d), [0, 0.0])
            a[0] += int(sel.sum())
            a[1] += float(w[sel].sum())
    return {k: (v[0], v[1]) for k, v in acc.items()}


def build(merged_manifest: dict, cms: dict, *, prefer: str | None = None,
          scan: bool = True) -> dict:
    out: dict[str, dict] = {}
    for sample, info in sorted(merged_manifest.items()):
        datasets = info.get("datasets") or {}
        if not datasets:
            print(f"[meta] WARNING {sample}: no dataset_id map — re-merge with a version "
                  f"that writes it, or this sample cannot be normalised")
            continue
        n_ds = len(datasets)
        per_id: dict[int, tuple[int, float]] = {}
        if scan and n_ds > 1:
            files = [f for f in (info.get("files") or []) if os.path.exists(f)]
            if files:
                print(f"[meta] {sample}: scanning dataset_id over {len(files)} file(s) ...",
                      flush=True)
                per_id = scan_per_dataset(files)
            if not per_id:
                print(f"[meta] WARNING {sample}: could not scan per-dataset counts; "
                      f"nevents will be left null rather than guessed")
        for dsid, dataset in sorted(datasets.items(), key=lambda kv: int(kv[0])):
            nick, cms_entry = _match(cms, dataset, prefer)
            if nick is None:
                print(f"[meta] WARNING {dataset}: no CMS entry — xsec unavailable")
            xsec = cms_entry.get("xsec")
            if xsec is None:
                print(f"[meta] WARNING {dataset}: CMS entry {nick} has xsec=null")

            # Per-sample totals are all we have when a sample spans several datasets; the
            # split is NOT recoverable from the manifest alone, so say so rather than
            # dividing by n and inventing a number.
            shared = n_ds > 1
            if shared and int(dsid) in per_id:
                n_ev, sw = per_id[int(dsid)]
                shared = False                  # measured per dataset, not shared
            else:
                n_ev, sw = info.get("events"), info.get("sum_genweight")
            entry = {
                "nick": f"{dataset}_Delphes",
                "dbs": cms_entry.get("dbs"),
                "era": cms_entry.get("era"),
                "sample_type": cms_entry.get("sample_type"),
                "xsec": xsec,
                "cms_nick": nick,
                # --- ours, over the PROCESSED shard set ---
                "nevents": None if shared else n_ev,
                "generator_weight": (None if shared or not n_ev else sw / n_ev),
                "sum_genweight": None if shared else sw,
                "nfiles": len(info.get("files") or []),
                "shards": info.get("shards"),
                "shards_planned": info.get("shards_planned"),
                # --- Delphes provenance ---
                "delphes_sample": sample,
                "dataset_id": int(dsid),
                "subtree": info.get("subtree"),
                "maps_sha": info.get("maps_sha"),
            }
            if shared:
                entry["note"] = (
                    f"'{sample}' spans {n_ds} primary datasets; per-dataset nevents and "
                    f"generator_weight must be summed from the dataset_id column of the "
                    f"merged files (the manifest holds only the sample total "
                    f"{info.get('events')} events, sum_genweight "
                    f"{info.get('sum_genweight')})")
            out[f'{entry["nick"]}_id{dsid}' if n_ds > 1 else entry["nick"]] = entry
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--merged", required=True, help="merged dir holding manifest.json")
    ap.add_argument("--cms-datasets", required=True, help="the CMS datasets.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-scan", action="store_true",
                    help="skip the dataset_id scan; multi-dataset samples then carry "
                         "null nevents rather than measured per-dataset counts")
    ap.add_argument("--prefer", default="RunIII2024Summer24NanoAODv15",
                    help="substring preferred when several CMS campaigns share a dataset")
    args = ap.parse_args(argv)

    with open(os.path.join(args.merged, "manifest.json")) as fh:
        merged = json.load(fh)
    meta = build(merged, _load_cms(args.cms_datasets), prefer=args.prefer,
                 scan=not args.no_scan)

    out = args.out or os.path.join(args.merged, "datasets_delphes.json")
    with open(out, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    print(f"\n{'dataset':52s} {'xsec [pb]':>12s} {'nevents':>12s} {'gen_w':>8s}")
    for k, e in sorted(meta.items()):
        x = f"{e['xsec']:.6g}" if e.get("xsec") is not None else "MISSING"
        n = f"{e['nevents']:,}" if e.get("nevents") is not None else "per-id"
        g = f"{e['generator_weight']:.5f}" if e.get("generator_weight") is not None else "per-id"
        print(f"{k[:52]:52s} {x:>12s} {n:>12s} {g:>8s}")
    print(f"\n[meta] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
