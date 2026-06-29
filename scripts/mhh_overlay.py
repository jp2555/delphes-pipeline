"""Overlay reco m_HH — Delphes fast-sim vs CMS NanoAOD — per κ_λ point.

    pixi run python scripts/mhh_overlay.py --config config.yml \
        --delphes-dir /ceph/jpan/cms_nanoaod_2024_hh2b2tau/delphes \
        --nano-dir    /ceph/jpan/cms_nanoaod_2024_hh2b2tau \
        --out plots/mhh_overlay --max-events 20000

For each κ_λ sample it reconstructs m_HH = (two highest-b-tag jets) + (FastMTT di-τ) on both
the Delphes sample and the CMS NanoAOD (τ_h from the DeepTau ``Tau`` collection there), and
overlays the *normalized* m_HH lineshapes so the κ_λ-discriminating shape can be compared
detector-to-detector. Reuses the framework reconstruction so it matches the Level-4 gate.
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
from delphes_pipeline.validation.level4_kappa import _coll_p4_sum, _mass, _reco_mhh
from delphes_pipeline.validation.run_validation import load_config

_KL = re.compile(r"kl-(m?\d+p\d+)")


def _kl_label(path: str):
    m = _KL.search(os.path.basename(os.path.normpath(path)))
    return m.group(1) if m else None


def _ditau_p4(cand, met_x, met_y):
    """FastMTT di-τ 4-vector for the leading pair of a τ-candidate collection (any sample)."""
    sel = ak.to_numpy(ak.num(cand) >= 2)
    pair = cand[sel][ak.argsort(cand[sel].pt, axis=1, ascending=False)][:, :2]
    leg1, leg2 = _leg(pair[:, 0]), _leg(pair[:, 1])
    _, x1, x2 = fastmtt_mass(leg1, leg2, met_x[sel], met_y[sel], with_x=True)

    def tau(leg, x):
        nu = np.sqrt(leg["px"] ** 2 + leg["py"] ** 2 + leg["pz"] ** 2) * (1.0 - x) / x
        return leg["px"] / x, leg["py"] / x, leg["pz"] / x, leg["e"] + nu

    a, b = tau(leg1, x1), tau(leg2, x2)
    return {"px": a[0] + b[0], "py": a[1] + b[1], "pz": a[2] + b[2], "e": a[3] + b[3]}, sel


def _nano_mhh(nano):
    """reco m_HH on NanoAOD: two highest-b-tag jets + FastMTT di-τ (DeepTau Taus + e/μ)."""
    jets = nano.jets
    bb = jets[ak.argsort(jets.pt, axis=1, ascending=False, stable=True)]
    bb = bb[ak.argsort(bb.btag, axis=1, ascending=False, stable=True)][:, :2]
    n_jet = ak.to_numpy(ak.num(bb))
    bx, by, bz, be = _coll_p4_sum(bb)

    taus = nano.taus[nano.taus.vsjet >= nano.deeptau_medium()]
    th = ak.zip({"pt": taus.pt, "eta": taus.eta, "phi": taus.phi, "mass": taus.mass,
                 "is_tauh": ak.ones_like(taus.pt)})
    lep = lambda c: ak.zip({"pt": c.pt, "eta": c.eta, "phi": c.phi,
                            "mass": ak.zeros_like(c.pt), "is_tauh": ak.zeros_like(c.pt)})
    cand = ak.concatenate([th, lep(nano.electrons), lep(nano.muons)], axis=1)
    met_x = ak.to_numpy(ak.fill_none(nano.met.met * np.cos(nano.met.phi), np.nan))
    met_y = ak.to_numpy(ak.fill_none(nano.met.met * np.sin(nano.met.phi), np.nan))

    ditau, sel = _ditau_p4(cand, met_x, met_y)
    n = nano.n
    dpx, dpy, dpz, de = (np.full(n, np.nan) for _ in range(4))
    dpx[sel], dpy[sel], dpz[sel], de[sel] = ditau["px"], ditau["py"], ditau["pz"], ditau["e"]
    m = _mass(bx + dpx, by + dpy, bz + dpz, be + de)
    m[~((n_jet >= 2) & np.isfinite(dpx))] = np.nan
    return m[np.isfinite(m)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Delphes vs CMS NanoAOD m_HH overlay per κ_λ")
    ap.add_argument("--config", required=True, help="for the NanoAOD branches + b-tag/DeepTau WPs")
    ap.add_argument("--delphes-dir", required=True)
    ap.add_argument("--nano-dir", required=True)
    ap.add_argument("--out", default="plots/mhh_overlay")
    ap.add_argument("--max-events", type=int, default=20000)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    from delphes_pipeline.tuning.anchor import _resolve_wp
    wp = _resolve_wp(cfg.get("anchor", {}).get("wp", {}))
    branches = cfg.get("anchor", {}).get("branches")
    os.makedirs(args.out, exist_ok=True)
    bins = np.linspace(200, 900, 36)

    nano_by_kl = {}
    for d in glob.glob(os.path.join(args.nano_dir, "*kl-*")):
        kl = _kl_label(d)
        if kl and "NanoAOD" in d:
            nano_by_kl.setdefault(kl, d)

    for d in sorted(glob.glob(os.path.join(args.delphes_dir, "*kl-*"))):
        kl = _kl_label(d)
        if kl is None or kl not in nano_by_kl:
            print(f"[skip] {os.path.basename(d)}: no matching NanoAOD")
            continue
        print(f"[kl {kl}] reconstructing m_HH ...", flush=True)
        dm, _, _ = _reco_mhh(DelphesEvents(d, entry_stop=args.max_events))
        dm = dm[np.isfinite(dm)]
        nm = _nano_mhh(NanoAODEvents(nano_by_kl[kl], branches=branches, wp=wp, entry_stop=args.max_events))

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.hist(dm, bins=bins, density=True, histtype="step", lw=2, label=f"Delphes ({dm.size})")
        ax.hist(nm, bins=bins, density=True, histtype="step", lw=2, label=f"CMS NanoAOD ({nm.size})")
        ax.set_xlabel("reco $m_{HH}$ [GeV]"); ax.set_ylabel("a.u."); ax.set_title(f"$\\kappa_\\lambda$ = {kl}")
        ax.legend()
        out = os.path.join(args.out, f"mhh_{kl}.png")
        fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
        print(f"[kl {kl}] Delphes med {np.median(dm):.0f} / NanoAOD med {np.median(nm):.0f} GeV -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
