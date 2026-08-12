"""Compare two tuning-map sets — e.g. signal-derived vs tt̄-derived — map by map.

    pixi run python scripts/compare_maps.py \
        cards/tuning/maps_v1.json cards/tuning/maps_ttbar_v1.json \
        --labels signal ttbar --out plots/maps_compare

Every map in ``maps_v1.json`` is measured on the HH signal, on both sides, and was until
recently applied to tt̄ as well. Whether that transfer is legitimate is an empirical
question, and this answers it: derive a second set on tt̄ and see how far apart they are.

The verdict column is a means, not an end. A map that agrees within its own statistical
error transfers; one that does not has to be derived per process, and the NSBI training
must then use the right set per sample (``convert.py --tuning-maps``). ``--tol`` sets the
relative difference above which a map is called out; the default 5% matches the level-0
closure tolerance, so a map that fails here would also fail its own closure if the other
process's value were imposed on it.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# scalar maps carry a single value at centers=[0.0] rather than a pT curve
_SCALAR_X = "overall"


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    return d["maps"], d.get("provenance", {})


def _curve(m):
    return (np.asarray(m.get("centers", []), dtype=float),
            np.asarray(m.get("values", []), dtype=float))


def compare(a, b, *, tol):
    """Per-map relative difference of b vs a, on a's grid. Returns rows for reporting."""
    rows = []
    for q in sorted(set(a) & set(b)):
        ca, va = _curve(a[q])
        cb, vb = _curve(b[q])
        if va.size == 0 or vb.size == 0:
            rows.append((q, float("nan"), float("nan"), "empty on one side"))
            continue
        if a[q].get("x") == _SCALAR_X or ca.size == 1 or cb.size == 1:
            rel = abs(vb[0] - va[0]) / abs(va[0]) if va[0] else float("nan")
            rows.append((q, rel, float("nan"), _verdict(rel, tol)))
            continue
        # interpolate b onto a's grid; flat outside, matching TuningMaps.efficiency
        vb_on_a = np.interp(ca, cb, vb, left=vb[0], right=vb[-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.abs(vb_on_a - va) / np.abs(va)
        r = r[np.isfinite(r)]
        if r.size == 0:
            rows.append((q, float("nan"), float("nan"), "no overlap"))
            continue
        rows.append((q, float(r.max()), float(ca[int(np.nanargmax(np.abs(vb_on_a - va) / np.abs(va)))]),
                     _verdict(float(r.max()), tol)))
    return rows


def _verdict(rel, tol):
    if not np.isfinite(rel):
        return "?"
    return "TRANSFERS" if rel <= tol else "PER-PROCESS"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("maps_a")
    ap.add_argument("maps_b")
    ap.add_argument("--labels", nargs=2, default=("A", "B"))
    ap.add_argument("--out", default="plots/maps_compare")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="relative difference above which a map needs per-process derivation")
    args = ap.parse_args(argv)

    a, pa = load(args.maps_a)
    b, pb = load(args.maps_b)
    la, lb = args.labels
    print(f"[compare] {la}: {args.maps_a}  ({pa.get('anchor_nanoaod', '?')})")
    print(f"[compare] {lb}: {args.maps_b}  ({pb.get('anchor_nanoaod', '?')})")
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a:
        print(f"[compare] only in {la}: {', '.join(only_a)}")
    if only_b:
        print(f"[compare] only in {lb}: {', '.join(only_b)}")

    rows = compare(a, b, tol=args.tol)
    print(f"\n{'map':22s} {'max |rel diff|':>14s} {'at pT':>8s}   verdict")
    for q, rel, at, verdict in rows:
        at_s = f"{at:8.0f}" if np.isfinite(at) else f"{'scalar':>8s}"
        rel_s = f"{rel*100:13.1f}%" if np.isfinite(rel) else f"{'n/a':>14s}"
        print(f"{q:22s} {rel_s} {at_s}   {verdict}")
    need = [q for q, rel, _, v in rows if v == "PER-PROCESS"]
    print(f"\n[compare] {len(need)}/{len(rows)} maps exceed {args.tol*100:.0f}%"
          + (f": {', '.join(need)}" if need else ""))
    if need:
        print("[compare] -> derive per process and ntuplize each sample with its own "
              "--tuning-maps; a single map set biases the NSBI likelihood ratio.")

    plot = [q for q in sorted(set(a) & set(b)) if _curve(a[q])[0].size > 1]
    if plot:
        nrow = (len(plot) + 3) // 4
        fig, axes = plt.subplots(nrow, 4, figsize=(18, 3.6 * nrow), squeeze=False)
        for ax in axes.flat[len(plot):]:
            ax.axis("off")
        for ax, q in zip(axes.flat, plot):
            for m, lab in ((a[q], la), (b[q], lb)):
                c, v = _curve(m)
                ax.plot(c, v, marker="o", ms=3, label=lab)
            ax.set_title(q, fontsize=10)
            ax.set_xlabel("pT [GeV]")
            ax.legend(fontsize=8)
        fig.suptitle(f"tuning maps: {la} vs {lb} — a map that does not overlap "
                     "must be derived per process")
        fig.tight_layout()
        os.makedirs(args.out, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.out, f"maps_compare.{ext}"), dpi=130)
        print(f"[compare] wrote {args.out}/maps_compare.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
