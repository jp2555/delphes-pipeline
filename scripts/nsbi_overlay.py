"""Overlay the 10 NSBI features — TUNED Delphes vs CMS NanoAOD — per κ_λ point.

    pixi run python scripts/nsbi_overlay.py --config config.yml \
        --delphes-dir /ceph/jpan/cms_nanoaod_2024_hh2b2tau/delphes \
        --nano-dir    /ceph/jpan/cms_nanoaod_2024_hh2b2tau \
        --out plots/nsbi_overlay --max-events 20000 [--tuned/--no-tuned] [--tautau-only]

Features: {mHH, cosθ*, pHH_T, mbb, ΔR_bb, mττ, ΔR_ττ, Δφ_HH, pH1_T, pH2_T}. The Delphes side
applies the tuning_maps (b-tag/τ_h re-tag + energy scale) by default so it is the *tuned*
sample that feeds the NSBI — that is the right thing to compare with CMS; ``--no-tuned`` shows
the stock card. Splitting the bb side (mbb, ΔR_bb) from the ττ side (mττ, ΔR_ττ) localizes any
Delphes-vs-CMS mismatch. NB: verify the cosθ* definition matches your NSBI's.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import awkward as ak
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.core.nanoaod import NanoAODEvents
from delphes_pipeline.extensions.mtautau import _leg, fastmtt_mass
from delphes_pipeline.validation.run_validation import load_config

_KL = re.compile(r"kl-(m?\d+p\d+)")
_FEATURES = ["mHH", "cosThetaStar", "pHH_T", "mbb", "dR_bb", "mtautau", "dR_tautau",
             "dphi_HH", "pH1_T", "pH2_T"]
_RANGES = {"mHH": (200, 900), "cosThetaStar": (0, 1), "pHH_T": (0, 300), "mbb": (0, 250),
           "dR_bb": (0, 5), "mtautau": (0, 250), "dR_tautau": (0, 5), "dphi_HH": (0, 3.2),
           "pH1_T": (0, 400), "pH2_T": (0, 400)}


def _kl(path):
    m = _KL.search(os.path.basename(os.path.normpath(path)))
    return m.group(1) if m else None


def _p4(pt, eta, phi, mass):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return {"px": px, "py": py, "pz": pz, "e": np.sqrt(px * px + py * py + pz * pz + mass * mass)}


def _add(a, b):
    return {k: a[k] + b[k] for k in a}


def _mass(p):
    return np.sqrt(np.maximum(p["e"] ** 2 - p["px"] ** 2 - p["py"] ** 2 - p["pz"] ** 2, 0.0))


def _pt(p):
    return np.hypot(p["px"], p["py"])


def _eta(p):
    pt = _pt(p)
    return np.arcsinh(np.divide(p["pz"], pt, out=np.zeros_like(pt), where=pt > 0))


def _phi(p):
    return np.arctan2(p["py"], p["px"])


def _dphi(a, b):
    return np.abs((_phi(a) - _phi(b) + np.pi) % (2 * np.pi) - np.pi)


def _dR(a, b):
    return np.hypot(_eta(a) - _eta(b), _dphi(a, b))


def _cos_theta_star(h1, hh):
    """|cosθ*| of H1 in the HH rest frame (polar angle vs the beam)."""
    bx, by, bz = hh["px"] / hh["e"], hh["py"] / hh["e"], hh["pz"] / hh["e"]
    b2 = np.clip(bx * bx + by * by + bz * bz, 0, 0.999999)
    gamma = 1.0 / np.sqrt(1 - b2)
    bp = bx * h1["px"] + by * h1["py"] + bz * h1["pz"]
    fac = np.where(b2 > 0, (gamma - 1) / np.where(b2 > 0, b2, 1), 0.0)
    pz = h1["pz"] + fac * bp * bz - gamma * bz * h1["e"]
    px = h1["px"] + fac * bp * bx - gamma * bx * h1["e"]
    py = h1["py"] + fac * bp * by - gamma * by * h1["e"]
    p = np.sqrt(px * px + py * py + pz * pz)
    return np.abs(np.divide(pz, p, out=np.zeros_like(p), where=p > 0))


def _tau_cands_delphes(ev):
    from delphes_pipeline.validation.level1_candles.selections import tau_candidates
    return tau_candidates(ev)


def _tau_cands_nano(ev):
    taus = ev.taus[ev.taus.vsjet >= ev.deeptau_medium()]
    th = ak.zip({"pt": taus.pt, "eta": taus.eta, "phi": taus.phi, "mass": taus.mass,
                 "is_tauh": ak.ones_like(taus.pt)})
    lep = lambda c: ak.zip({"pt": c.pt, "eta": c.eta, "phi": c.phi,
                            "mass": ak.zeros_like(c.pt), "is_tauh": ak.zeros_like(c.pt)})
    return ak.concatenate([th, lep(ev.electrons), lep(ev.muons)], axis=1)


def features(ev, *, nano, tautau_only=False, mtautau_min=20.0):
    """The 10 NSBI features over events with a reconstructed bb + di-τ system.

    ``mtautau_min`` drops the m_ττ≈0 spike (a FastMTT failure / a collinear or
    double-counted τ-pair) so it doesn't distort the shape normalization.
    """
    jets = ev.jets
    bsrc = jets if nano else jets[jets.tautag == 0]                 # τ_h are separate on NanoAOD
    bb = bsrc[ak.argsort(bsrc.pt, axis=1, ascending=False, stable=True)]
    bb = bb[ak.argsort(bb.btag, axis=1, ascending=False, stable=True)][:, :2]
    cand = _tau_cands_nano(ev) if nano else _tau_cands_delphes(ev)
    cand = cand[ak.argsort(cand.pt, axis=1, ascending=False, stable=True)]

    sel = ak.to_numpy((ak.num(bb) >= 2) & (ak.num(cand) >= 2))
    bb, cand = bb[sel], cand[sel][:, :2]
    met = ev.met[sel]
    met_x = ak.to_numpy(ak.fill_none(met.met * np.cos(met.phi), np.nan))
    met_y = ak.to_numpy(ak.fill_none(met.met * np.sin(met.phi), np.nan))

    b1 = _p4(*(ak.to_numpy(bb[:, 0][k]) for k in ("pt", "eta", "phi", "mass")))
    b2 = _p4(*(ak.to_numpy(bb[:, 1][k]) for k in ("pt", "eta", "phi", "mass")))
    t1 = _p4(*(ak.to_numpy(cand[:, 0][k]) for k in ("pt", "eta", "phi", "mass")))
    t2 = _p4(*(ak.to_numpy(cand[:, 1][k]) for k in ("pt", "eta", "phi", "mass")))
    n_th = ak.to_numpy(ak.sum(cand.is_tauh, axis=1))

    leg1, leg2 = _leg(cand[:, 0]), _leg(cand[:, 1])
    _, x1, x2 = fastmtt_mass(leg1, leg2, met_x, met_y, with_x=True)

    def tau_full(leg, x):
        nu = np.sqrt(leg["px"] ** 2 + leg["py"] ** 2 + leg["pz"] ** 2) * (1 - x) / x
        return {"px": leg["px"] / x, "py": leg["py"] / x, "pz": leg["pz"] / x, "e": leg["e"] + nu}

    H1, H2 = _add(b1, b2), _add(tau_full(leg1, x1), tau_full(leg2, x2))
    HH = _add(H1, H2)
    pH1, pH2 = _pt(H1), _pt(H2)
    out = {
        "mHH": _mass(HH), "cosThetaStar": _cos_theta_star(H1, HH), "pHH_T": _pt(HH),
        "mbb": _mass(H1), "dR_bb": _dR(b1, b2), "mtautau": _mass(H2), "dR_tautau": _dR(t1, t2),
        "dphi_HH": _dphi(H1, H2), "pH1_T": np.maximum(pH1, pH2), "pH2_T": np.minimum(pH1, pH2),
    }
    keep = np.isfinite(out["mHH"]) & (out["mtautau"] > mtautau_min)
    if tautau_only:
        keep = keep & (n_th == 2)
    return {k: v[keep] for k, v in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NSBI 10-feature overlay: tuned Delphes vs CMS NanoAOD")
    ap.add_argument("--config", required=True)
    ap.add_argument("--delphes-dir", required=True)
    ap.add_argument("--nano-dir", required=True)
    ap.add_argument("--out", default="plots/nsbi_overlay")
    ap.add_argument("--max-events", type=int, default=20000)
    ap.add_argument("--tuned", dest="tuned", action="store_true", default=True)
    ap.add_argument("--no-tuned", dest="tuned", action="store_false")
    ap.add_argument("--tautau-only", action="store_true")
    ap.add_argument("--mtautau-min", type=float, default=20.0, help="drop the m_ττ≈0 spike below this (GeV)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    from delphes_pipeline.tuning.anchor import _resolve_wp
    wp = _resolve_wp(cfg.get("anchor", {}).get("wp", {}))
    branches = cfg.get("anchor", {}).get("branches")
    maps_path = cfg.get("tuning_maps") if args.tuned else None
    tuning = None
    if maps_path:
        from delphes_pipeline.tuning.maps import TuningMaps
        tuning = TuningMaps.load(maps_path)
        print(f"[overlay] applying tuning maps from {maps_path}")
    os.makedirs(args.out, exist_ok=True)

    nano_by_kl = {_kl(d): d for d in glob.glob(os.path.join(args.nano_dir, "*kl-*")) if _kl(d) and "NanoAOD" in d}

    for d in sorted(glob.glob(os.path.join(args.delphes_dir, "*kl-*"))):
        kl = _kl(d)
        if kl is None or kl not in nano_by_kl:
            continue
        print(f"[kl {kl}] reconstructing features ...", flush=True)
        dev = DelphesEvents(d, entry_stop=args.max_events)
        if tuning is not None:
            from delphes_pipeline.tuning.maps import RetaggedEvents
            dev = RetaggedEvents(dev, tuning, np.random.default_rng(0))
        df = features(dev, nano=False, tautau_only=args.tautau_only, mtautau_min=args.mtautau_min)
        nf = features(NanoAODEvents(nano_by_kl[kl], branches=branches, wp=wp, entry_stop=args.max_events),
                      nano=True, tautau_only=args.tautau_only, mtautau_min=args.mtautau_min)

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for ax, feat in zip(axes.flat, _FEATURES):
            lo, hi = _RANGES[feat]
            b = np.linspace(lo, hi, 41)
            centres = 0.5 * (b[:-1] + b[1:])
            for data, lab in ((df[feat], "Delphes"), (nf[feat], "NanoAOD")):
                d = data[(data >= lo) & (data <= hi)]          # normalize over the plotted range
                if d.size:
                    h, _ = np.histogram(d, bins=b, density=True)
                    ax.step(centres, h, where="mid", lw=2, label=f"{lab} ({d.size})")
            ax.set_xlabel(feat); ax.legend(fontsize=8)
        fig.suptitle(f"$\\kappa_\\lambda$ = {kl}" + ("  (tuned)" if tuning is not None else "  (stock)"))
        out = os.path.join(args.out, f"nsbi_{kl}.png")
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        print(f"[kl {kl}] -> {out}  (Delphes {df['mHH'].size}, NanoAOD {nf['mHH'].size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
