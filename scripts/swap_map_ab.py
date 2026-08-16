"""Measure the per-process map artifact directly, by swapping the map set.

    pixi run python scripts/swap_map_ab.py \
        --delphes '/ceph/jpan/gen-delphes/*TTto2L2Nu*Delphes_v1' \
        --maps-own cards/tuning/maps_ttbar_v1.json \
        --maps-other cards/tuning/maps_v1.json \
        --out plots/swap_ab --max-events 200000

Per-process maps are applied per process today: signal through maps_v1, ttbar through
maps_ttbar_v1. A signal/background likelihood ratio then inherits the DIFFERENCE BETWEEN
THE MAPS as learned S/B shape, which lands on the measured parameter. This is the only
direct measurement of how large that is, and it must be run on v1 before v1 is retired to
a private cross-check.

Method: the SAME events, the SAME rng seed, processed twice with two map sets. The seed
matters — the correction is stochastic (tag draws, response and mass resampling, MET
smearing), so an unpaired comparison would fold RNG noise into the answer. With one seed
the draws are identical and only the thresholds and quantile grids differ, so every
per-feature difference is attributable to the maps.

The number to read is the per-feature total-variation distance. It is an upper bound on
what a classifier could exploit as if it were physics: a feature with TVD ~ 0 cannot carry
the artifact, one with large TVD can.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsbi_overlay import _FEATURES, _DIAG, _RANGES, _tvd, features  # noqa: E402

from delphes_pipeline.core.io import DelphesEvents  # noqa: E402
from delphes_pipeline.tuning.maps import RetaggedEvents, TuningMaps  # noqa: E402


def run(delphes, maps_own, maps_other, *, max_events, seed=0, sel_kw=None):
    sel_kw = sel_kw or {}
    out = {}
    for label, path in (("own", maps_own), ("other", maps_other)):
        ev = DelphesEvents(delphes, entry_stop=max_events)
        # one seed for both arms: pairs the stochastic draws so the difference is the maps
        view = RetaggedEvents(ev, TuningMaps.load(path), np.random.default_rng(seed))
        print(f"[swap] {label}: {os.path.basename(path)} -> "
              f"{', '.join(sorted(view.retagged_fields))}", flush=True)
        out[label] = features(view, nano=False, **sel_kw)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--delphes", required=True)
    ap.add_argument("--maps-own", required=True, help="the map set this process normally gets")
    ap.add_argument("--maps-other", required=True, help="the map set from the OTHER process")
    ap.add_argument("--out", default="plots/swap_ab")
    ap.add_argument("--max-events", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diagnostics", action="store_true")
    args = ap.parse_args(argv)

    res = run(args.delphes, args.maps_own, args.maps_other, max_events=args.max_events,
              seed=args.seed)
    a, b = res["own"], res["other"]
    feats = _FEATURES + (_DIAG if args.diagnostics else [])

    sep = {f: _tvd(a[f], b[f], *_RANGES[f]) for f in feats}
    print(f"\n[swap] per-process map artifact — own vs other map set, same events, "
          f"same seed\n{'feature':16s} {'TVD':>8s}   {'n own':>9s} {'n other':>9s}")
    for f in sorted(feats, key=lambda x: -sep[x]):
        print(f"{f:16s} {sep[f]:8.4f}   {a[f].size:9,} {b[f].size:9,}")
    worst = max(sep, key=lambda f: sep[f])
    print(f"\n[swap] largest artifact: {worst} (TVD {sep[worst]:.4f})")
    print(f"[swap] selection moved {a['mHH'].size:,} -> {b['mHH'].size:,} events "
          f"({100 * (b['mHH'].size / max(a['mHH'].size, 1) - 1):+.1f}%)")
    print("[swap] this is an UPPER BOUND on what a classifier could pick up as physics; "
          "it is the size of the C1 cross-process sculpting term.")

    os.makedirs(args.out, exist_ok=True)
    nrow = (len(feats) + 4) // 5
    fig, axes = plt.subplots(nrow, 5, figsize=(20, 3.6 * nrow))
    for ax in axes.flat[len(feats):]:
        ax.axis("off")
    for ax, f in zip(axes.flat, feats):
        lo, hi = _RANGES[f]
        bins = np.linspace(lo, hi, 41)
        c = 0.5 * (bins[:-1] + bins[1:])
        for d, lab in ((a[f], "own maps"), (b[f], "other process's maps")):
            d = d[(d >= lo) & (d <= hi)]
            if d.size:
                h, _ = np.histogram(d, bins=bins, density=True)
                ax.step(c, h, where="mid", lw=2, label=lab)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f"{f}   (TVD {sep[f]:.3f})")
        ax.legend(fontsize=8)
    fig.suptitle("swap-map A/B: same events and seed, two map sets — "
                 "the gap is what a S/B ratio would learn as shape")
    fig.tight_layout()
    p = os.path.join(args.out, "swap_ab.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"[swap] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
