"""Does one CMS τ response map serve every process, or is one needed per process?

    pixi run python scripts/check_anchor_transfer.py \
        --labels HH ttbar \
        --anchors '/ceph/jpan/cms_nanoaod_2024_hh2b2tau/*kl-1p00*NanoAODv15-PowhegBugFix*' \
                  '/ceph/jpan/cms_nanoaod_2024_hh2b2tau/TTto2L2Nu*NanoAODv15-150X*' \
        --config config.v1.yml --out plots/anchor_transfer

Option B draws the τ energy from the anchor's response quantiles. That is only legitimate
across processes if the CMS response really is a detector property rather than a
final-state one. The response is reco/gen for objects passing the same ID, so it *should*
transfer — but tt̄ is a busier final state than HH, and if isolation-correlated ID
efficiency or nearby activity shifts the measured response, it will not.

PASS means one map serves the campaign. FAIL means the map must be derived per process and
each sample ntuplized with its own ``--tuning-maps``. The criteria are deliberately tight
where it matters: medians to 1% (they set the scale, which enters m_ττ linearly) and the
upper quantile ratios to 5% (they are the shape option B exists to reproduce).
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.nanoaod import NanoAODEvents
from delphes_pipeline.tuning import anchor as A
from delphes_pipeline.validation.run_validation import load_config

_MED_TOL = 0.01      # scale: enters m_tautau linearly
_SHAPE_TOL = 0.05    # tail: the thing a multiplicative map cannot fix
_REPORT_Q = (0.75, 0.90, 0.95, 0.99)


def _profile(path, cfg, *, max_events, fake):
    ac = cfg.get("anchor", {})
    nano = NanoAODEvents(path, branches=ac.get("branches"),
                         wp=A._resolve_wp(ac.get("wp", {})), entry_stop=max_events)
    bins = obs.DEFAULT_PT_BINS
    prof = (A._nano_tau_fake_response(nano, bins) if fake
            else A._nano_tau_energy_response(nano, bins))
    return nano.n, prof


def _rows(prof):
    """(center, median, {q: q/median}) per populated pT bin."""
    lv = np.asarray((prof.aux or {}).get("quantile_levels", []), dtype=float)
    qv = np.asarray((prof.aux or {}).get("quantile_values", []), dtype=float)
    out = []
    for c, row in zip(np.asarray(prof.centers, dtype=float), qv):
        med = float(np.interp(0.5, lv, row))
        if med <= 0:
            continue
        out.append((float(c), med, {q: float(np.interp(q, lv, row)) / med for q in _REPORT_Q}))
    return out


def compare(a_rows, b_rows):
    """Pair bins by nearest center; report the two deltas that decide transfer."""
    verdicts = []
    for ca, ma, qa in a_rows:
        if not b_rows:
            continue
        cb, mb, qb = min(b_rows, key=lambda r: abs(r[0] - ca))
        d_med = abs(mb / ma - 1.0)
        d_shape = max(abs(qb[q] / qa[q] - 1.0) for q in _REPORT_Q)
        ok = d_med <= _MED_TOL and d_shape <= _SHAPE_TOL
        verdicts.append((ca, ma, mb, d_med, d_shape, ok))
    return verdicts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--anchors", nargs=2, required=True)
    ap.add_argument("--labels", nargs=2, default=("A", "B"))
    ap.add_argument("--max-events", type=int, default=200000)
    ap.add_argument("--fake", action="store_true",
                    help="compare the FAKE response (gen-unmatched Tau vs GenJet) instead")
    ap.add_argument("--out", default="plots/anchor_transfer")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    la, lb = args.labels
    kind = "fake" if args.fake else "real"
    na, pa = _profile(args.anchors[0], cfg, max_events=args.max_events, fake=args.fake)
    nb, pb = _profile(args.anchors[1], cfg, max_events=args.max_events, fake=args.fake)
    ra, rb = _rows(pa), _rows(pb)
    print(f"[transfer] {kind} tau response: {la} ({na} ev, {len(ra)} bins) vs "
          f"{lb} ({nb} ev, {len(rb)} bins)")
    if not ra or not rb:
        print("[transfer] one side has no populated bins — cannot conclude")
        return 1

    v = compare(ra, rb)
    print(f"\n{'pT':>7s} {la+' med':>10s} {lb+' med':>10s} {'d med':>8s} "
          f"{'d shape':>9s}   verdict")
    for c, ma, mb, dm, ds, ok in v:
        print(f"{c:7.0f} {ma:10.4f} {mb:10.4f} {dm*100:7.2f}% {ds*100:8.2f}%   "
              f"{'ok' if ok else 'DIFFERS'}")
    bad = [x for x in v if not x[5]]
    print(f"\n[transfer] {len(bad)}/{len(v)} bins exceed "
          f"({_MED_TOL*100:.0f}% median, {_SHAPE_TOL*100:.0f}% shape)")
    if bad:
        print(f"[transfer] -> derive the {kind} response per process and ntuplize each "
              "sample with its own --tuning-maps")
    else:
        print(f"[transfer] -> one {kind} response map serves both; transfer is justified")

    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for lab, rows in ((la, ra), (lb, rb)):
        axes[0].plot([r[0] for r in rows], [r[1] for r in rows], marker="o", label=lab)
        axes[1].plot([r[0] for r in rows], [r[2][0.95] for r in rows], marker="o", label=lab)
    axes[0].set_xlabel("gen pT [GeV]"), axes[0].set_ylabel("median response")
    axes[1].set_xlabel("gen pT [GeV]"), axes[1].set_ylabel("q95 / median (shape)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(f"CMS {kind} tau response: {la} vs {lb} — left = scale, right = shape")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"anchor_transfer_{kind}.{ext}"), dpi=130)
    print(f"[transfer] wrote {args.out}/anchor_transfer_{kind}.pdf")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
