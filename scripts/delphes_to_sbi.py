"""Merged Delphes ntuples -> the SBI input format the NSBI notebook reads.

    pixi run python scripts/delphes_to_sbi.py \
        --ntuple /ceph/jpan/ntuples_untuned/merged \
        --sample signal --out saved_datasets/dihiggs_delphes_data.root

Target format: one ROOT file per process, one TTree each, flat float64 branches --
identical to what ``convert_powheg_to_sbi.py`` produces from CMS NanoAOD:

    signal  dihiggs_delphes_data.root : tree_sbi_lam0, tree_sbi_lam1, tree_sbi_lam5
    ttbar   ttbar_delphes_data.root   : tree_ttbar
    DY      dy_delphes_data.root      : tree_dy

Required branches:
    m_hh, m_bb, m_tautau, pt_hh, deta_hh, dr_bb, dr_tautau,
    cos_theta_star, dphi_hh, pt_h1, pt_h2, weights

**The definitions here are the CMS side's, not this repo's overlay's, and several
differ.** The point of the exercise is a Delphes baseline comparable with the NSBI
result already obtained on CMS NanoAOD, so a mismatched definition would silently
compare two different measurements:

  * ``m_tautau`` is the VISIBLE di-tau mass (``m_vis``). It is NOT FastMTT -- the CMS
    converter maps ``m_vis -> m_tautau`` and carries FastMTT only as an optional extra.
  * ``cos_theta_star`` is ``tanh(deta_hh / 2)``, not the Collins-Soper polar angle in
    the HH rest frame that ``scripts/nsbi_overlay.py`` computes.
  * ``pt_h1`` is the bb system and ``pt_h2`` the tautau system -- fixed by content, NOT
    sorted by pT (the overlay uses max/min).
  * ``m_hh`` / ``pt_hh`` are built from bb + VISIBLE tautau, so they are not the
    FastMTT-corrected mHH either.
  * ``weights`` is genWeight (x puweight on CMS; Delphes has no pileup, so 1).

Channel: the CMS NSBI test used ``mt`` and ``et`` only -- one light lepton plus one
tau_h. Leg 1 is the lepton, leg 2 the tau_h. This applies the same requirement, so the
tau_h-tau_h and fully-leptonic channels are excluded here as they were there.
"""

from __future__ import annotations

import argparse
import os

import awkward as ak
import numpy as np

from delphes_pipeline.core.io import NtupleEvents, available_kl

REQUIRED = ["m_hh", "m_bb", "m_tautau", "pt_hh", "deta_hh", "dr_bb", "dr_tautau",
            "cos_theta_star", "dphi_hh", "pt_h1", "pt_h2", "weights"]

# tree name per process, matching config_powheg.yml's Tree field
TREES = {"ttbar": "tree_ttbar", "dy": "tree_dy",
         "singletop": "tree_singletop", "wjets": "tree_wjets"}


def _p4(pt, eta, phi, mass):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return px, py, pz, np.sqrt(px * px + py * py + pz * pz + mass * mass)


def _sum_p4(a, b):
    """(pt, eta, phi, mass) of the sum of two (pt, eta, phi, mass) legs."""
    p1, p2 = _p4(*a), _p4(*b)
    px, py, pz, e = (p1[i] + p2[i] for i in range(4))
    pt = np.hypot(px, py)
    p = np.sqrt(px * px + py * py + pz * pz)
    eta = np.arctanh(np.clip(pz / np.maximum(p, 1e-10), -1 + 1e-10, 1 - 1e-10))
    return pt, eta, np.arctan2(py, px), np.sqrt(np.maximum(e * e - p * p, 0.0))


def _dphi(a, b):
    d = np.abs(a - b)
    return np.where(d > np.pi, 2 * np.pi - d, d)


def _dR(eta1, phi1, eta2, phi2):
    return np.hypot(eta1 - eta2, _dphi(phi1, phi2))


def select(ev, *, lep_pt_min=20.0, lep_eta_max=2.4,
           tau_pt_min=20.0, tau_eta_max=2.3,
           jet_pt_min=20.0, jet_eta_max=2.4, clean_dr=0.4):
    """The mt/et selection: one light lepton, one tau_h, two b-tagged jets.

    Jets are cleaned against BOTH selected legs before the b pair is chosen. On Delphes a
    tau_h *is* an AK4 jet, so without that removal the tau would be free to enter the b
    pair -- an artefact that lands on dr_bb and m_bb.
    """
    e, m = ev.electrons, ev.muons
    lep = ak.concatenate([
        ak.zip({"pt": e.pt, "eta": e.eta, "phi": e.phi,
                "mass": ak.zeros_like(e.pt)}),
        ak.zip({"pt": m.pt, "eta": m.eta, "phi": m.phi,
                "mass": ak.zeros_like(m.pt)}),
    ], axis=1)
    lep = lep[(lep.pt > lep_pt_min) & (np.abs(lep.eta) <= lep_eta_max)]
    lep = lep[ak.argsort(lep.pt, axis=1, ascending=False, stable=True)]

    j = ev.jets
    tau = j[(j.tautag == 1) & (j.pt > tau_pt_min) & (np.abs(j.eta) <= tau_eta_max)]
    tau = tau[ak.argsort(tau.pt, axis=1, ascending=False, stable=True)]

    keep = ak.to_numpy((ak.num(lep) >= 1) & (ak.num(tau) >= 1))
    lep, tau, j = lep[keep][:, :1], tau[keep][:, :1], j[keep]

    j = j[(j.pt > jet_pt_min) & (np.abs(j.eta) <= jet_eta_max)]
    far = (_dR(j.eta, j.phi, lep[:, 0].eta, lep[:, 0].phi) > clean_dr) & \
          (_dR(j.eta, j.phi, tau[:, 0].eta, tau[:, 0].phi) > clean_dr)
    j = j[far]
    # two hardest b-tagged jets, falling back to the hardest jets, as the CMS bpair does
    j = j[ak.argsort(j.pt, axis=1, ascending=False, stable=True)]
    bb = j[ak.argsort(j.btag, axis=1, ascending=False, stable=True)][:, :2]

    ok = ak.to_numpy(ak.num(bb) >= 2)
    idx = np.flatnonzero(keep)[ok]
    return lep[ok][:, 0], tau[ok][:, 0], bb[ok], idx


def features(ev):
    """The 12 required branches, using the CMS converter's definitions."""
    lep, tau, bb, idx = select(ev)
    g = lambda c, k: ak.to_numpy(c[k])                       # noqa: E731
    b1 = (g(bb[:, 0], "pt"), g(bb[:, 0], "eta"), g(bb[:, 0], "phi"), g(bb[:, 0], "mass"))
    b2 = (g(bb[:, 1], "pt"), g(bb[:, 1], "eta"), g(bb[:, 1], "phi"), g(bb[:, 1], "mass"))
    l1 = (g(lep, "pt"), g(lep, "eta"), g(lep, "phi"), g(lep, "mass"))
    l2 = (g(tau, "pt"), g(tau, "eta"), g(tau, "phi"), g(tau, "mass"))

    hbb = _sum_p4(b1, b2)
    htt = _sum_p4(l1, l2)                                     # VISIBLE, no FastMTT
    hh = _sum_p4(hbb, htt)

    deta = np.abs(hbb[1] - htt[1])
    # Ladder-weight rule: lepton_sf is part of the EVENT WEIGHT, not a correction some
    # methods apply and others don't. A weight column consumed by the ratio training and
    # not by the histogram baseline silently breaks N >= D >= H by construction. It is
    # 1.0 on an untuned sample, so this is a no-op for the baseline and correct for v2.
    w = ak.to_numpy(ev.array["genWeight"])[idx].astype(np.float64)
    if "lepton_sf" in ak.fields(ev.array):
        w = w * ak.to_numpy(ev.array["lepton_sf"])[idx].astype(np.float64)
    out = {
        "m_hh": hh[3], "pt_hh": hh[0],
        "m_bb": hbb[3], "m_tautau": htt[3],
        "dr_bb": _dR(b1[1], b1[2], b2[1], b2[2]),
        "dr_tautau": _dR(l1[1], l1[2], l2[1], l2[2]),
        "deta_hh": deta,
        "dphi_hh": _dphi(hbb[2], htt[2]),
        # tanh(deta/2), NOT the boosted-frame polar angle
        "cos_theta_star": np.tanh((hbb[1] - htt[1]) / 2.0),
        "pt_h1": hbb[0],            # bb by CONTENT, not the harder Higgs
        "pt_h2": htt[0],
        "weights": w,
    }
    out = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
    good = np.ones(len(out["m_hh"]), dtype=bool)
    for v in out.values():
        good &= np.isfinite(v)
    return {k: v[good] for k, v in out.items()}, int((~good).sum())


def _report(name, d, dropped):
    n = len(d["m_hh"])
    w = d["weights"]
    neg = int((w < 0).sum())
    print(f"  {name}: {n:,} events ({dropped:,} dropped non-finite), "
          f"sum(w)={w.sum():.1f}" + (f", {neg:,} negative weights" if neg else ""))
    missing = [b for b in REQUIRED if b not in d]
    if missing:
        raise SystemExit(f"[sbi] missing required branches: {missing}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ntuple", required=True, help="merged ntuple dir or glob")
    ap.add_argument("--sample", required=True,
                    help="'signal' (one tree per kl) or a background name (ttbar, dy, ...)")
    ap.add_argument("--out", required=True, help="output ROOT file")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--tree", default=None, help="override the output tree name")
    args = ap.parse_args(argv)

    import uproot

    trees = {}
    if args.sample == "signal":
        for kl in available_kl(args.ntuple):
            ev = NtupleEvents(args.ntuple, kl=kl, entry_stop=args.max_events)
            d, dropped = features(ev)
            name = (f"tree_sbi_lam{int(kl)}" if kl == int(kl)
                    else "tree_sbi_lam" + str(kl).replace(".", "p"))
            _report(f"{name} (kl={kl:g}, {ev.n:,} read)", d, dropped)
            trees[name] = d
    else:
        ev = NtupleEvents(args.ntuple, entry_stop=args.max_events)
        d, dropped = features(ev)
        name = args.tree or TREES.get(args.sample, f"tree_{args.sample}")
        _report(f"{name} ({ev.n:,} read)", d, dropped)
        trees[name] = d

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with uproot.recreate(args.out) as fh:
        for name, d in trees.items():
            fh[name] = d
    print(f"[sbi] wrote {args.out}: {', '.join(sorted(trees))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
