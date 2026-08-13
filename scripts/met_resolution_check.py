"""Is the Delphes pT_miss RESOLUTION anisotropic and ΣE_T-dependent, as CMS's is?

    pixi run python scripts/met_resolution_check.py --config config.v1.yml \
        --delphes-root '/ceph/jpan/gen-delphes/*kl-1p00*Delphes_v1' \
        --nano-path   '/ceph/jpan/cms_nanoaod_2024_hh2b2tau/*kl-1p00*NanoAODv15-PowhegBugFix*' \
        --out plots/met_resolution --max-events 20000

``met_smear`` adds ISOTROPIC Gaussian noise of one FLAT width, chosen so the overall
pT_miss resolution lands on the anchor's. The real resolution is neither: it is larger
along the hadronic recoil than across it, and it grows with event activity. Since
m_ττ = m_vis/sqrt(x1 x2) is NONLINEAR in the fitted energy fractions, excess noise does
not merely broaden m_ττ — it drags the mean up too. That fits both remaining symptoms
(Delphes m_ττ high AND broad) and is what this measures.

Method, applied identically to both tiers: take the residual R = pT_miss(reco) −
pT_miss(gen), project it along the di-τ axis and across it, and report the WIDTH of each
(the propagation check measured their MEANS — the response; this is the resolution).
Two questions:

  ANISOTROPY  sigma_para / sigma_perp.  CMS should exceed 1; if Delphes sits at 1 the
              smearing is round where the data is not.
  ACTIVITY    sigma versus HT.  CMS should rise; a flat Delphes means the map cannot
              follow it, and the residual will be process-dependent — which matters
              because tt̄ and DY are far busier than the HH signal it was tuned on.

HT is recomputed here from JETS on both tiers with a common acceptance, deliberately NOT
from the stored ScalarHT/sumEt: those are defined differently on the two tiers (see
``anchor._nano_met_resolution``), so binning by them would compare different variables.
"""

from __future__ import annotations

import argparse
import os
import sys

import awkward as ak
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from met_propagation_check import _project, _ETA_MAX, _PT_MIN, _DR  # noqa: E402

from delphes_pipeline.core import observables as obs  # noqa: E402
from delphes_pipeline.core.io import DelphesEvents  # noqa: E402
from delphes_pipeline.validation.run_validation import load_config  # noqa: E402

_HT_BINS = np.array([0, 120, 180, 250, 350, 500, 900])
_HT_PT, _HT_ETA = 20.0, 4.7


def _ht(jets):
    """Event HT from jets, one definition for both tiers."""
    j = jets[(jets.pt > _HT_PT) & (np.abs(jets.eta) <= _HT_ETA)]
    return ak.to_numpy(ak.sum(j.pt, axis=1))


def _width(x):
    """Robust half 16-84 spread — the residual has tails that would distort an RMS."""
    if x.size < 25:
        return float("nan"), float("nan")
    lo, hi = np.percentile(x, [16, 84])
    w = 0.5 * (hi - lo)
    return float(w), float(w / np.sqrt(2.0 * x.size))      # ~sigma/sqrt(2N)


def measure(ev, *, nano):
    if nano:
        taus = ev.taus[ev.taus.vsjet >= ev.deeptau_medium()]
        cand = taus[(taus.pt > _PT_MIN) & (np.abs(taus.eta) <= _ETA_MAX)]
        vis = ev.genvistau
    else:
        j = ev.jets
        cand = j[(j.tautag == 1) & (j.pt > _PT_MIN) & (np.abs(j.eta) <= _ETA_MAX)]
        vis = obs.gen_visible_taus(ev.gen, dr=_DR)
    cand = cand[ak.argsort(cand.pt, axis=1, ascending=False, stable=True)][:, :2]
    keep = ak.to_numpy(ak.num(cand) >= 2)
    ht = _ht(ev.jets)[keep]
    _, para, perp, ht = _project(cand[keep], vis[keep], ev.met[keep], ev.genmet[keep],
                                 extra=ht)
    return para, perp, ht


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--delphes-root", required=True)
    ap.add_argument("--nano-path", required=True)
    ap.add_argument("--out", default="plots/met_resolution")
    ap.add_argument("--max-events", type=int, default=20000)
    ap.add_argument("--no-tuned", action="store_true",
                    help="skip the tuning maps (met_smear is the thing under test)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    dev = DelphesEvents(args.delphes_root,
                        treename=cfg.get("input", {}).get("treename", "Delphes"),
                        entry_stop=args.max_events)
    if not args.no_tuned and cfg.get("tuning_maps"):
        from delphes_pipeline.tuning.maps import RetaggedEvents, TuningMaps
        dev = RetaggedEvents(dev, TuningMaps.load(cfg["tuning_maps"]), np.random.default_rng(0))
        print(f"[res] Delphes corrections: {', '.join(sorted(dev.retagged_fields))}")
    from delphes_pipeline.core.nanoaod import NanoAODEvents
    from delphes_pipeline.tuning import anchor as A
    ac = cfg.get("anchor", {})
    nev = NanoAODEvents(args.nano_path, branches=ac.get("branches"),
                        wp=A._resolve_wp(ac.get("wp", {})), entry_stop=args.max_events)

    out = {}
    for lbl, ev, nano in (("Delphes", dev, False), ("CMS", nev, True)):
        out[lbl] = measure(ev, nano=nano)
        para, perp, ht = out[lbl]
        wpa, epa = _width(para)
        wpe, epe = _width(perp)
        print(f"\n[res] {lbl}: {para.size} pairs")
        print(f"       sigma parallel to di-tau   {wpa:6.2f} +- {epa:.2f} GeV")
        print(f"       sigma perpendicular        {wpe:6.2f} +- {epe:.2f} GeV")
        print(f"       ANISOTROPY para/perp       {wpa/wpe:6.3f}")

    print(f"\n{'HT bin':>12s}  " + "  ".join(f"{l+' para':>12s}" for l in out) + "   ratio")
    curves = {l: ([], [], []) for l in out}
    for lo, hi in zip(_HT_BINS[:-1], _HT_BINS[1:]):
        vals = {}
        for lbl, (para, _, ht) in out.items():
            m = (ht >= lo) & (ht < hi)
            w, e = _width(para[m])
            vals[lbl] = (w, e)
            if np.isfinite(w):
                curves[lbl][0].append(0.5 * (lo + hi)); curves[lbl][1].append(w); curves[lbl][2].append(e)
        d, c = vals.get("Delphes", (np.nan,))[0], vals.get("CMS", (np.nan,))[0]
        r = f"{d/c:6.3f}" if np.isfinite(d) and np.isfinite(c) and c else "     -"
        cells = "  ".join(f"{vals[l][0]:12.2f}" if np.isfinite(vals[l][0]) else f"{'-':>12s}"
                          for l in out)
        print(f"{f'{lo}-{hi}':>12s}  {cells}   {r}")

    for lbl in out:
        x, y = np.array(curves[lbl][0]), np.array(curves[lbl][1])
        if x.size > 1:
            print(f"[res] {lbl:8s} d(sigma)/d(HT) = {np.polyfit(x, y, 1)[0]*100:+.2f} GeV per 100 GeV HT")

    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for lbl in out:
        para, perp, _ = out[lbl]
        axes[0].hist(para, bins=60, range=(-80, 80), histtype="step", density=True,
                     label=f"{lbl} para")
        axes[0].hist(perp, bins=60, range=(-80, 80), histtype="step", ls="--", density=True,
                     label=f"{lbl} perp")
        x, y, e = (np.array(v) for v in curves[lbl])
        if x.size:
            axes[1].errorbar(x, y, yerr=e, marker="o", capsize=2, label=lbl)
    axes[0].set_xlabel(r"$R$ projection [GeV]"), axes[0].set_ylabel("normalised")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("HT [GeV]"), axes[1].set_ylabel(r"$\sigma(R\cdot\hat u)$ [GeV]")
    axes[1].legend(fontsize=8)
    fig.suptitle("pT_miss resolution: solid = along the di-tau axis, dashed = across; "
                 "right = growth with activity")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"met_resolution.{e}"), dpi=130)
    print(f"\n[res] wrote {args.out}/met_resolution.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
