"""Is the Delphes τ energy excess MISSING energy, or just cone bookkeeping?

    pixi run python scripts/met_propagation_check.py --config config.v1.yml \
        --delphes-root '/ceph/jpan/gen-delphes/*kl-1p00*Delphes_v1' \
        --out plots/met_propagation --max-events 20000

Option B redraws the τ pT downward. Whether pT_miss must follow decides ~1/3 of the
m_ττ correction, and the two answers come from two different pictures:

  * MISMEASURED — the detector lost that energy. Then it belongs in pT_miss, and not
    propagating leaves the event unbalanced.
  * COME DEFINITION — an R=0.4 cone simply swept up underlying event that a narrow HPS
    cone would have left as unclustered. pT_miss is MINUS THE SUM OF ALL VISIBLE ENERGY,
    and that UE is already in the sum, so re-labelling it changes nothing. Propagating
    would then move energy that never went missing, fabricating pT_miss.

This measures which. Define, per event, over the SELECTED τ pair:

    D = Σ (reco τ vec − gen-visible τ vec)      the excess attributed to the τ
    R = pT_miss(reco) vec − pT_miss(gen) vec    what pT_miss actually got wrong

and project R on the direction of D. The two pictures predict opposite things:

    missing-energy   ->  R·û(D) ≈ −|D|, slope ≈ −1 against |D|
    cone-definition  ->  R·û(D) ≈ 0,    slope ≈ 0, uncorrelated with |D|

The perpendicular projection R·v̂ is the control: it must be flat and centred on zero
under BOTH pictures, so a non-zero slope there means the estimator, not the physics.
Run on RAW (untuned) Delphes — the question is about what the simulation did, not about
what our maps then did to it.
"""

from __future__ import annotations

import argparse
import os

import awkward as ak
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.core.matching import nearest_target_fields
from delphes_pipeline.validation.run_validation import load_config

_PT_MIN, _ETA_MAX, _DR = 20.0, 2.3, 0.4


def _xy(pt, phi):
    return pt * np.cos(phi), pt * np.sin(phi)


def measure(ev, *, nano=False):
    """(|D|, R·û(D), R·v̂(D)) per event with a gen-matched τ_hτ_h pair.

    ``nano=True`` runs the identical construction on the CMS anchor, where the τ come
    from the ``Tau`` collection at Medium DeepTau and the gen reference is ``GenVisTau``.
    That comparison is the one that matters: an offset present on BOTH tiers cancels in
    the Delphes-vs-CMS comparison, while one present only on Delphes is a real response
    bias feeding FastMTT — and ``met_smear``, being pure noise, cannot remove it.
    """
    if nano:
        taus = ev.taus[ev.taus.vsjet >= ev.deeptau_medium()]
        cand = taus[(taus.pt > _PT_MIN) & (np.abs(taus.eta) <= _ETA_MAX)]
        cand = cand[ak.argsort(cand.pt, axis=1, ascending=False, stable=True)][:, :2]
        keep = ak.to_numpy(ak.num(cand) >= 2)
        return _project(cand[keep], ev.genvistau[keep], ev.met[keep], ev.genmet[keep])
    jets = ev.jets
    cand = jets[(jets.tautag == 1) & (jets.pt > _PT_MIN) & (np.abs(jets.eta) <= _ETA_MAX)]
    cand = cand[ak.argsort(cand.pt, axis=1, ascending=False, stable=True)][:, :2]
    keep = ak.to_numpy(ak.num(cand) >= 2)
    return _project(cand[keep], obs.gen_visible_taus(ev.gen, dr=_DR)[keep],
                    ev.met[keep], ev.genmet[keep])


def _project(cand, vis, met, gmet, extra=None):
    """Returns (|D|, R.u, R.v) and, when ``extra`` is given, that array aligned to the
    same surviving events — so a caller can bin the projections by anything it likes."""
    ok, ref = nearest_target_fields(cand, vis, _DR, ("pt", "phi"))
    n = len(cand)
    ok = ok.reshape(n, 2)
    gpt = np.nan_to_num(ref["pt"], nan=0.0).reshape(n, 2)
    gphi = np.nan_to_num(ref["phi"], nan=0.0).reshape(n, 2)
    rpt = ak.to_numpy(cand.pt).reshape(n, 2)
    rphi = ak.to_numpy(cand.phi).reshape(n, 2)

    both = ok.all(axis=1) & (gpt > 0).all(axis=1)     # both legs gen-matched
    rx, ry = _xy(rpt, rphi)
    gx, gy = _xy(gpt, gphi)
    dx = (rx - gx).sum(axis=1)[both]
    dy = (ry - gy).sum(axis=1)[both]

    mx, my = _xy(ak.to_numpy(ak.fill_none(met.met, 0.0)),
                 ak.to_numpy(ak.fill_none(met.phi, 0.0)))
    gmx, gmy = _xy(ak.to_numpy(ak.fill_none(gmet.met, 0.0)),
                   ak.to_numpy(ak.fill_none(gmet.phi, 0.0)))
    rx_, ry_ = (mx - gmx)[both], (my - gmy)[both]

    d = np.hypot(dx, dy)
    good = d > 1e-6
    ux, uy = dx[good] / d[good], dy[good] / d[good]
    para = rx_[good] * ux + ry_[good] * uy
    perp = -rx_[good] * uy + ry_[good] * ux
    if extra is None:
        return d[good], para, perp
    return d[good], para, perp, np.asarray(extra)[both][good]


def _profile(x, y, edges):
    c, m, e = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() < 20:
            continue
        c.append(float(x[s].mean())); m.append(float(np.mean(y[s])))
        e.append(float(np.std(y[s]) / np.sqrt(s.sum())))
    return np.array(c), np.array(m), np.array(e)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--delphes-root")
    ap.add_argument("--nano-path", help="run on the CMS anchor instead, for comparison")
    ap.add_argument("--out", default="plots/met_propagation")
    ap.add_argument("--max-events", type=int, default=20000)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.nano_path:
        from delphes_pipeline.core.nanoaod import NanoAODEvents
        from delphes_pipeline.tuning import anchor as A
        ac = cfg.get("anchor", {})
        ev = NanoAODEvents(args.nano_path, branches=ac.get("branches"),
                           wp=A._resolve_wp(ac.get("wp", {})), entry_stop=args.max_events)
        d, para, perp = measure(ev, nano=True)
        tier = "CMS anchor"
    else:
        ev = DelphesEvents(args.delphes_root,
                           treename=cfg.get("input", {}).get("treename", "Delphes"),
                           entry_stop=args.max_events)
        d, para, perp = measure(ev)
        tier = "RAW Delphes"
    print(f"[met] {d.size} gen-matched tau_h tau_h pairs on {tier}")
    print(f"[met] tau excess |D|      mean {d.mean():7.2f}  median {np.median(d):7.2f} GeV")
    print(f"[met] R parallel to D     mean {para.mean():7.2f} +- {para.std()/np.sqrt(d.size):.2f} GeV")
    print(f"[met] R perpendicular     mean {perp.mean():7.2f} +- {perp.std()/np.sqrt(d.size):.2f} GeV  (control)")

    edges = np.array([0, 5, 10, 15, 20, 30, 45, 70, 120])
    cx, cy, ce = _profile(d, para, edges)
    px, py, pe = _profile(d, perp, edges)
    slope = np.polyfit(cx, cy, 1)[0] if cx.size > 1 else float("nan")
    slope_perp = np.polyfit(px, py, 1)[0] if px.size > 1 else float("nan")
    print(f"\n[met] slope d(R.u)/d|D| = {slope:+.3f}   (control, perpendicular: {slope_perp:+.3f})")
    print("[met] interpretation:")
    if abs(slope) < 0.2:
        print("      ~0  -> the excess is NOT missing energy. It is cone bookkeeping: the UE")
        print("            is already inside pT_miss's sum, so propagating would fabricate it.")
        print("      -> keep propagate_met OFF.")
    elif slope < -0.6:
        print("      ~-1 -> the excess IS energy pT_miss failed to account for.")
        print("      -> turn propagate_met ON.")
    else:
        print(f"      partial ({slope:+.2f}) -> propagate only that FRACTION, i.e. the")
        print("            fluctuation about the map median, not the whole offset.")

    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist2d(d, para, bins=(40, 40), range=((0, 100), (-80, 80)), cmap="Blues")
    axes[0].errorbar(cx, cy, yerr=ce, color="crimson", marker="o", lw=2, label="profile")
    for lbl, yy, st in (("missing-energy: slope -1", -edges, "--"), ("cone-definition: slope 0",
                                                                    np.zeros_like(edges), ":")):
        axes[0].plot(edges, yy, st, color="k", lw=1.5, label=lbl)
    axes[0].set_xlabel(r"$|D|$ = tau excess [GeV]"), axes[0].set_ylabel(r"$R\cdot\hat u(D)$ [GeV]")
    axes[0].set_ylim(-80, 80), axes[0].legend(fontsize=8)
    axes[1].errorbar(px, py, yerr=pe, color="grey", marker="s", lw=2)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xlabel(r"$|D|$ [GeV]"), axes[1].set_ylabel(r"$R\cdot\hat v$ [GeV] (control)")
    fig.suptitle("Is the Delphes tau energy excess missing energy? "
                 f"slope = {slope:+.3f} (control {slope_perp:+.3f})")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"met_propagation.{e}"), dpi=130)
    print(f"\n[met] wrote {args.out}/met_propagation.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
