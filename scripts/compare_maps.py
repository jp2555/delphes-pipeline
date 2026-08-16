"""Compare two tuning-map sets — e.g. signal-derived vs tt̄-derived — map by map.

    pixi run python scripts/compare_maps.py \
        cards/tuning/maps_v1.json cards/tuning/maps_ttbar_v1.json \
        --labels signal ttbar --out plots/maps_compare

Every map in ``maps_v1.json`` is measured on the HH signal, on both sides, and was until
recently applied to tt̄ as well. Whether that transfer is legitimate is an empirical
question, and this answers it: derive a second set on tt̄ and see how far apart they are.

The verdict column is a means, not an end -- but read the verdict the right way round.

Agreement within statistics validates the PARAMETERISATION: the map is an object-level
detector property and one merged set can be applied everywhere. Disagreement means a
variable is MISSING from the parameterisation (flavour, prong multiplicity, eta,
pileup) -- it is NOT a licence to apply a different map to each process.

That distinction is the whole point. A detector response is a property of the final
state: one forward model for everything. Per-process APPLICATION makes two kinematically
identical jets smear differently according to an unobservable process label, and a
signal/background likelihood ratio then inherits the difference BETWEEN the maps as
learned S/B shape. Deriving per process is a measurement strategy (each process is a
control region weighting the same universal map differently); applying per process is a
modelling error. See docs/tuning_for_nsbi_audit.md.

``--tol`` sets the relative difference above which a map is called out; the default 5%
matches the level-0 closure tolerance, so a map that fails here would also fail its own
closure if the other process's value were imposed on it.
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


def _sigma(m):
    """Per-bin uncertainty on a map's values, or NaN where it cannot be derived.

    E3's pass condition is that per-process determinations agree *within statistics* --
    not within a flat relative tolerance. The failing bins here are the highest-pT ones,
    which are also the emptiest, so a relative cut alone cannot separate "the
    parameterisation is missing a variable" from "this bin has 468 entries".
    """
    v = np.asarray(m.get("values", []), dtype=float)
    n = np.asarray(m.get("counts", []), dtype=float)
    if v.size == 0 or n.size != v.size:
        return np.full(v.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        qv = m.get("quantile_values")
        if qv:
            # a median drawn from a stored distribution: sigma_median = 1.253 sigma/sqrt(n)
            q = np.asarray(qv, dtype=float)
            if q.ndim == 2 and q.shape[0] == v.size:
                lo = np.percentile(q, 16, axis=1)
                hi = np.percentile(q, 84, axis=1)
                return 1.253 * (hi - lo) / 2.0 / np.sqrt(np.maximum(n, 1))
        if np.all((v >= 0) & (v <= 1)):
            return np.sqrt(np.maximum(v * (1 - v), 0) / np.maximum(n, 1))   # binomial
        if m.get("x") == "ht" or "resolution" in str(m.get("ylabel", "")):
            return v / np.sqrt(2 * np.maximum(n, 1))                        # width
    return np.full(v.shape, np.nan)


def compare(a, b, *, tol, nsigma=3.0):
    """Per-map relative difference of b vs a, on a's grid. Returns rows for reporting."""
    rows = []
    for q in sorted(set(a) & set(b)):
        ca, va = _curve(a[q])
        cb, vb = _curve(b[q])
        if va.size == 0 or vb.size == 0:
            rows.append((q, float("nan"), float("nan"), float("nan"), "empty on one side"))
            continue
        if a[q].get("x") == _SCALAR_X or ca.size == 1 or cb.size == 1:
            rel = abs(vb[0] - va[0]) / abs(va[0]) if va[0] else float("nan")
            rows.append((q, rel, float("nan"), float("nan"),
                         _verdict(rel, tol, float("nan"), nsigma)))
            continue
        # Same binning on both sides (DEFAULT_PT_BINS) with slightly different bin
        # CENTRES, since a centre is the mean pT in the bin. Compare bin-by-bin so the
        # per-bin counts line up; only fall back to interpolation if the grids differ,
        # and then without a pull, since the errors no longer correspond.
        if ca.size == cb.size:
            diff = np.abs(vb - va)
            sa, sb = _sigma(a[q]), _sigma(b[q])
            with np.errstate(divide="ignore", invalid="ignore"):
                pull = diff / np.sqrt(sa ** 2 + sb ** 2)
                r = diff / np.abs(va)
        else:
            vb_on_a = np.interp(ca, cb, vb, left=vb[0], right=vb[-1])
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.abs(vb_on_a - va) / np.abs(va)
            pull = np.full(r.shape, np.nan)
        if not np.isfinite(r).any():
            rows.append((q, float("nan"), float("nan"), float("nan"), "no overlap"))
            continue
        i = int(np.nanargmax(np.where(np.isfinite(r), r, -np.inf)))
        mx_pull = float(np.nanmax(pull)) if np.isfinite(pull).any() else float("nan")
        rows.append((q, float(r[i]), float(ca[i]), mx_pull,
                     _verdict(float(r[i]), tol, mx_pull, nsigma)))
    return rows


def _verdict(rel, tol, pull, nsigma):
    """Consistent within statistics is the question; the relative cut is the fallback.

    A verdict marked * had no usable per-bin error, so it rests on the relative
    tolerance alone and should not be read as a statistical statement.
    """
    if not np.isfinite(rel):
        return "?"
    if np.isfinite(pull):
        return "consistent" if pull <= nsigma else "DIFFERS"
    return "consistent*" if rel <= tol else "DIFFERS*"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("maps_a")
    ap.add_argument("maps_b")
    ap.add_argument("--labels", nargs=2, default=("A", "B"))
    ap.add_argument("--out", default="plots/maps_compare")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="fallback relative-difference cut, used only where no per-bin "
                         "error can be derived (verdict marked *)")
    ap.add_argument("--nsigma", type=float, default=3.0,
                    help="pull above which per-process determinations are called "
                         "inconsistent — E3's actual pass condition")
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

    rows = compare(a, b, tol=args.tol, nsigma=args.nsigma)
    print(f"\n{'map':22s} {'max |rel diff|':>14s} {'at pT':>8s} {'max pull':>9s}   verdict")
    for q, rel, at, pull, verdict in rows:
        at_s = f"{at:8.0f}" if np.isfinite(at) else f"{'scalar':>8s}"
        rel_s = f"{rel*100:13.1f}%" if np.isfinite(rel) else f"{'n/a':>14s}"
        pl_s = f"{pull:9.1f}" if np.isfinite(pull) else f"{'-':>9s}"
        print(f"{q:22s} {rel_s} {at_s} {pl_s}   {verdict}")
    need = [q for q, _, _, _, v in rows if v.startswith("DIFFERS")]
    print(f"\n[compare] {len(need)}/{len(rows)} maps differ beyond {args.nsigma:g}"
          f"\u03c3 (or, where * marks no usable error, beyond {args.tol*100:.0f}%)"
          + (f": {', '.join(need)}" if need else ""))
    if need:
        print("[compare] -> a map that DIFFERS means the parameterisation is missing a")
        print("[compare]    VARIABLE, not that the map should be applied per process.")
        print("[compare]    A detector response is a property of the final state: one")
        print("[compare]    forward model for every process. Applying a different map to")
        print("[compare]    signal and background makes the map DIFFERENCE into learned")
        print("[compare]    S/B shape, which lands directly on the measured parameter.")
        print("[compare]    Add the missing variable (flavour / gluon fraction, decay")
        print("[compare]    mode or prong count, eta, sum-Et), re-derive, merge to ONE")
        print("[compare]    frozen set, then ntuplize everything with it.")
        print("[compare]    See docs/tuning_for_nsbi_audit.md")

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
        fig.suptitle(f"tuning maps: {la} vs {lb} — curves that do not overlap mark a "
                     "MISSING VARIABLE in the parameterisation, not a per-process patch")
        fig.tight_layout()
        os.makedirs(args.out, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.out, f"maps_compare.{ext}"), dpi=130)
        print(f"[compare] wrote {args.out}/maps_compare.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
