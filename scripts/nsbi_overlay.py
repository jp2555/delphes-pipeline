"""Overlay the 10 NSBI features — TUNED Delphes vs CMS NanoAOD — per κ_λ point.

    pixi run python scripts/nsbi_overlay.py --config config.yml \
        --delphes-dir /ceph/jpan/cms_nanoaod_2024_hh2b2tau/delphes \
        --nano-dir    /ceph/jpan/cms_nanoaod_2024_hh2b2tau \
        --out plots/nsbi_overlay --max-events 20000 [--tuned/--no-tuned] [--tautau-only]
        [--no-clean]

Features: {mHH, cosθ*, pHH_T, mbb, ΔR_bb, mττ, ΔR_ττ, Δφ_HH, pH1_T, pH2_T}. The Delphes side
applies the tuning_maps (b-tag/τ_h re-tag + energy scale) by default so it is the *tuned*
sample that feeds the NSBI — that is the right thing to compare with CMS; ``--no-tuned`` shows
the stock card. Splitting the bb side (mbb, ΔR_bb) from the ττ side (mττ, ΔR_ττ) localizes any
Delphes-vs-CMS mismatch. NB: verify the cosθ* definition matches your NSBI's.

Both sides run the SAME selection (``--clean``, on by default): a common τ-candidate
acceptance (pT > 20, |eta| < 2.3), then pick the di-τ pair, then keep jets at pT/|eta|
acceptance and ΔR > 0.4 from both selected τ. Delphes τ_h *are* jets while
CMS keeps them in a separate ``Tau`` collection, so without a common overlap removal the two
sides build the bb pair from different jet pools — an artefact that lands on ΔR_bb.
``--no-clean`` restores the old asymmetric behaviour to size that effect.
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


def _kl_value(tag: str) -> float:
    """'0p00' -> 0.0, 'm2p50' -> -2.5. The merged ntuple stores kl as a number."""
    neg = tag.startswith("m")
    v = float(tag.lstrip("m").replace("p", "."))
    return -v if neg else v
_FEATURES = ["mHH", "cosThetaStar", "pHH_T", "mbb", "dR_bb", "mtautau", "dR_tautau",
             "dphi_HH", "pH1_T", "pH2_T"]
# diagnostic (not NSBI) observables: the FastMTT *inputs*. m_ττ = m_vis/√(x₁x₂) with the
# x fitted against pT_miss, so if the selection matches but m_ττ does not, the difference
# has to be in the τ four-vectors or in the MET — these panels separate the two.
_DIAG = ["mvis", "tau1_pt", "tau2_pt", "met"]

# The CMS Run-3 HH->bbtautau DNN input set (AN table 29), in the rotated frame it uses:
# every momentum is rotated by -phi(visible di-tau), so the tau pair lies at phi=0.
# Only the entries we actually overlay are listed here; CMS_DNN_NOT_OVERLAID records the
# rest with the reason, so the audit still accounts for all 26 inputs.
_CMS_OBJ = ["lep1", "lep2", "b1", "b2", "Htt", "Hbb", "Hbbtt"]
_CMS = ["met_px", "met_py"] + [f"{o}_{c}" for o in _CMS_OBJ for c in ("E", "px", "py", "pz")]

CMS_DNN_NOT_OVERLAID = [
    ("FatJet E/px/py/pz, FatJet exist, FatJet_tautau",
     "OUT OF SCOPE: we run the resolved category only, so the boosted AK8 inputs carry "
     "no information here. Not a Delphes limitation -- both tiers do expose an AK8 jet."),
    ("cov(MET) xx, xy, yy", "Delphes has no MET covariance; only an overall resolution. "
                            "A synthesised covariance would be an assumption, not a measurement."),
    ("ParticleNet / UParT score", "Delphes gives a binary tag bit, not a continuous "
                                  "discriminant; our re-tag reproduces the WP efficiency only."),
    ("HHbTag score", "a CMS-specific DNN over event-level inputs; no Delphes counterpart."),
    ("decay mode (lepton_1,2)", "Delphes has no tau substructure at all -- a tau_h is an "
                                "AK4 jet, so 1-prong / 1-prong+pi0 / 3-prong is undefined."),
    ("charge (lepton_1,2)", "Delphes jets carry an approximate jet charge, not a reconstructed "
                            "tau charge."),
    ("pair type", "we run the tau_h tau_h channel only; e-tau and mu-tau are not yet built."),
]
_RANGES = {"mHH": (200, 900), "cosThetaStar": (0, 1), "pHH_T": (0, 300), "mbb": (0, 250),
           "dR_bb": (0, 5), "mtautau": (0, 250), "dR_tautau": (0, 5),
           "dphi_HH": (0, np.pi),
           "pH1_T": (0, 400), "pH2_T": (0, 400),
           "mvis": (0, 200), "tau1_pt": (0, 200), "tau2_pt": (0, 150), "met": (0, 200)}


def _cms_range(name):
    """Plot range by component: Cartesian momenta are ~symmetric, energies positive."""
    if name.endswith("_E"):
        return (0, 400) if name.startswith(("lep", "b1", "b2")) else (0, 1200)
    if name.endswith("_pz"):
        return (-500, 500)
    return (-250, 250)          # px, py (including met_px/met_py)


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


def _visible_gen_taus(ev, nano):
    """The visible hadronic gen τ of the event, for gen-matching the selected pair.

    NanoAOD exposes them directly as ``GenVisTau``; on Delphes we take the hadronic gen
    τ (its visible products are collinear with it, so the direction is the same to well
    inside ΔR=0.4 — adequate for a match, though not for a pT comparison).
    """
    if nano:
        return ev.genvistau
    from delphes_pipeline.core import observables as obs
    return obs.gen_taus(ev.gen, hadronic_only=True)


def _rot(p, cos_a, sin_a):
    """Rotate a 4-vector dict about the beam axis by angle a (px,py only)."""
    return {"px": p["px"] * cos_a + p["py"] * sin_a,
            "py": -p["px"] * sin_a + p["py"] * cos_a,
            "pz": p["pz"], "e": p["e"]}


def features(ev, *, nano, tautau_only=False, mtautau_min=20.0, clean=True,
             jet_pt_min=20.0, jet_eta_max=2.4, clean_dr=0.4, with_match=False,
             tau_pt_min=20.0, tau_eta_max=2.3, cms_dnn=False):
    """The 10 NSBI features over events with a reconstructed bb + di-τ system.

    ``mtautau_min`` drops the m_ττ≈0 spike (a FastMTT failure / a collinear or
    double-counted τ-pair) so it doesn't distort the shape normalization.

    ``clean`` (default) applies the SAME selection to both sides, in the CMS order:
    pick the di-τ pair first, then keep jets with ``pt > jet_pt_min``,
    ``|eta| <= jet_eta_max`` and ``ΔR > clean_dr`` from *both* selected τ candidates.
    Without it the two sides are not comparable — the Delphes side used to drop every
    τ-tagged jet from the b pool while the NanoAOD side kept all jets (τ_h live in a
    separate ``Tau`` collection there, but their jets are still in ``Jet``), so a
    τ-jet could enter the CMS bb pair and not the Delphes one. That asymmetry lands
    squarely on ΔR_bb. ``clean=False`` restores the old behaviour for comparison.
    """
    cand = _tau_cands_nano(ev) if nano else _tau_cands_delphes(ev)
    if clean:
        # Common τ-candidate acceptance. Without it the two sides start from different
        # collections with different floors: a Delphes τ_h *is* a jet, so it exists down
        # to the card's JetPTMin (15 GeV) and out to TauEtaMax (2.5), while CMS NanoAOD
        # ``Tau`` begins near 18-20 GeV. Delphes then keeps soft/forward τ that CMS never
        # had — a low-pT-Higgs population, i.e. large ΔR_ττ and (wider opening angle) a
        # larger visible mass. pT>20, |η|<2.3 is also the analysis cut (AN-25-103 §4.6).
        cand = cand[(cand.pt > tau_pt_min) & (np.abs(cand.eta) <= tau_eta_max)]
    if tautau_only:
        cand = cand[cand.is_tauh == 1]        # τ_hτ_h channel: pick the 2 leading τ_h, not 2 of all
    cand = cand[ak.argsort(cand.pt, axis=1, ascending=False, stable=True)]

    jets = ev.jets
    if clean:
        jets = jets[(jets.pt > jet_pt_min) & (np.abs(jets.eta) <= jet_eta_max)]
    elif not nano:
        jets = jets[jets.tautag == 0]         # legacy: asymmetric, Delphes-only τ removal

    # the di-τ pair must exist before jets can be cleaned against it
    has_pair = ak.to_numpy(ak.num(cand) >= 2)
    vis_all = _visible_gen_taus(ev, nano)[has_pair] if with_match else None
    cand, jets, met_all = cand[has_pair][:, :2], jets[has_pair], ev.met[has_pair]
    if clean:
        from delphes_pipeline.core.matching import matched_to_any
        jets = jets[~matched_to_any(jets, cand, clean_dr)]

    bb = jets[ak.argsort(jets.pt, axis=1, ascending=False, stable=True)]
    bb = bb[ak.argsort(bb.btag, axis=1, ascending=False, stable=True)][:, :2]

    sel = ak.to_numpy(ak.num(bb) >= 2)
    bb, cand, met = bb[sel], cand[sel], met_all[sel]
    matched = None
    if with_match:
        from delphes_pipeline.core.matching import matched_to_any
        # "matched" = BOTH selected τ candidates sit on a real hadronic gen τ
        n_ok = ak.sum(matched_to_any(cand, vis_all[sel], 0.4), axis=1)
        matched = ak.to_numpy(n_ok >= 2)
    met_x = ak.to_numpy(ak.fill_none(met.met * np.cos(met.phi), np.nan))
    met_y = ak.to_numpy(ak.fill_none(met.met * np.sin(met.phi), np.nan))

    b1 = _p4(*(ak.to_numpy(bb[:, 0][k]) for k in ("pt", "eta", "phi", "mass")))
    b2 = _p4(*(ak.to_numpy(bb[:, 1][k]) for k in ("pt", "eta", "phi", "mass")))
    t1 = _p4(*(ak.to_numpy(cand[:, 0][k]) for k in ("pt", "eta", "phi", "mass")))
    t2 = _p4(*(ak.to_numpy(cand[:, 1][k]) for k in ("pt", "eta", "phi", "mass")))

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
        # FastMTT inputs, for diagnosing an m_ττ shift that survives a symmetric selection.
        # m_vis is the decisive one: m_ττ = m_vis/√(x₁x₂) factorises, so comparing the
        # VISIBLE di-τ mass — no FastMTT at all — splits an m_ττ offset in two. If m_vis is
        # already off by the same fraction the τ four-vector scale is wrong (tau_escale);
        # if m_vis agrees and only m_ττ does not, the τ energies are right and the error is
        # in the x fit, i.e. MET. The two have disjoint fixes, so measure before tuning.
        "mvis": _mass(_add(t1, t2)),
        "tau1_pt": _pt(t1), "tau2_pt": _pt(t2), "met": np.hypot(met_x, met_y),
    }
    if cms_dnn:
        # the CMS DNN frame: rotate everything so the visible di-τ sits at φ=0
        vis = _add(t1, t2)
        a = np.arctan2(vis["py"], vis["px"])
        ca, sa = np.cos(a), np.sin(a)
        objs = {"lep1": t1, "lep2": t2, "b1": b1, "b2": b2,
                "Htt": H2, "Hbb": H1, "Hbbtt": HH}
        for name, p in objs.items():
            r = _rot(p, ca, sa)
            out[f"{name}_E"] = r["e"]
            for c in ("px", "py", "pz"):
                out[f"{name}_{c}"] = r[c]
        out["met_px"] = met_x * ca + met_y * sa
        out["met_py"] = -met_x * sa + met_y * ca
    keep = np.isfinite(out["mHH"]) & (out["mtautau"] > mtautau_min)
    out = {k: v[keep] for k, v in out.items()}
    return (out, matched[keep]) if with_match else out


def _split_figure(kl, df, dm, nf, nm, args, tuning):
    """Per side: gen-matched vs fake τ pairs, each normalised to the side's TOTAL.

    Normalising both components to the same total (rather than each to unity) means the
    area under the fake curve IS the fake fraction, so shape and contamination are
    readable at once — the question being whether the Delphes ΔR_ττ excess lives in the
    fake component.
    """
    fd, fn = 1.0 - dm.mean(), 1.0 - nm.mean()
    print(f"[kl {kl}] fake (non-gen-matched) fraction: Delphes {fd:.3f}  CMS {fn:.3f}"
          f"   [{(~dm).sum()}/{dm.size} vs {(~nm).sum()}/{nm.size}]", flush=True)
    feats = _FEATURES + (_DIAG if args.diagnostics else [])
    nrow = (len(feats) + 4) // 5
    fig, axes = plt.subplots(nrow, 5, figsize=(20, 4 * nrow))
    for ax in axes.flat[len(feats):]:
        ax.axis("off")
    for ax, feat in zip(axes.flat, feats):
        lo, hi = _RANGES[feat]
        b = np.linspace(lo, hi, 41)
        centres = 0.5 * (b[:-1] + b[1:])
        for data, mask, colour, lab in ((df[feat], dm, "tab:blue", "Delphes"),
                                        (nf[feat], nm, "tab:orange", "CMS")):
            inr = (data >= lo) & (data <= hi)
            n_tot = max(int(inr.sum()), 1)
            for keep, style, tag in ((mask, "-", "gen-matched"), (~mask, "--", "fake")):
                d = data[inr & keep]
                if not d.size:
                    continue
                h, _ = np.histogram(d, bins=b)
                # density of this component relative to the side's full sample
                h = h / (n_tot * (b[1] - b[0]))
                ax.step(centres, h, where="mid", lw=1.8, ls=style, color=colour,
                        label=f"{lab} {tag} ({d.size})")
        ax.set_ylim(bottom=0)
        ax.set_xlabel(feat); ax.legend(fontsize=6)
    fig.suptitle(f"$\\kappa_\\lambda$ = {kl}  ·  gen-matched (solid) vs fake (dashed)"
                 + ("  · tuned" if tuning is not None else "  · stock"))
    out = os.path.join(args.out, f"split_{kl}.png")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[kl {kl}] -> {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NSBI 10-feature overlay: tuned Delphes vs CMS NanoAOD")
    ap.add_argument("--config", required=True)
    ap.add_argument("--delphes-dir")
    ap.add_argument("--ntuple", metavar="DIR_OR_GLOB",
                    help="overlay the MERGED ntuples instead of raw Delphes. The tuning "
                         "is already baked into them, so --tuned/--no-tuned does not "
                         "apply and the kl point is read from the 'kl' column.")
    ap.add_argument("--nano-dir", required=True)
    ap.add_argument("--out", default="plots/nsbi_overlay")
    ap.add_argument("--max-events", type=int, default=20000)
    ap.add_argument("--tuned", dest="tuned", action="store_true", default=True)
    ap.add_argument("--no-tuned", dest="tuned", action="store_false")
    ap.add_argument("--tautau-only", action="store_true")
    ap.add_argument("--mtautau-min", type=float, default=20.0, help="drop the m_ττ≈0 spike below this (GeV)")
    ap.add_argument("--clean", dest="clean", action="store_true", default=True,
                    help="symmetric jet cleaning on both sides (default)")
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="legacy asymmetric selection (Delphes drops τ-tagged jets, NanoAOD keeps all)")
    ap.add_argument("--jet-pt-min", type=float, default=20.0, help="common jet pT acceptance")
    ap.add_argument("--jet-eta-max", type=float, default=2.4, help="common jet |eta| acceptance")
    ap.add_argument("--clean-dr", type=float, default=0.4, help="ΔR(jet, selected τ) overlap removal")
    ap.add_argument("--tau-pt-min", type=float, default=20.0, help="common τ candidate pT acceptance")
    ap.add_argument("--tau-eta-max", type=float, default=2.3, help="common τ candidate |eta| acceptance")
    ap.add_argument("--cms-dnn", action="store_true",
                    help="overlay the CMS Run-3 DNN input set (rotated frame) instead of "
                         "the 10 NSBI features, and print which inputs Delphes cannot supply")
    ap.add_argument("--diagnostics", action="store_true",
                    help="also plot the FastMTT inputs (τ pT, MET) alongside the NSBI features")
    ap.add_argument("--split-gen-matched", action="store_true",
                    help="split each side into gen-matched vs fake τ pairs (diagnostic)")
    args = ap.parse_args(argv)

    if not (args.ntuple or args.delphes_dir):
        ap.error("one of --delphes-dir or --ntuple is required")
    if args.ntuple and not args.tuned:
        # the merged ntuple was written through the maps; there is no untuned view of
        # it, and silently ignoring --no-tuned would misreport what was plotted
        ap.error("--no-tuned cannot apply to --ntuple: the tuning is baked in")

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

    if args.cms_dnn:
        print("\n[cms-dnn] CMS Run-3 DNN inputs NOT overlaid:")
        for name, why in CMS_DNN_NOT_OVERLAID:
            print(f"    - {name:28s} {why}")
        print(f"\n[cms-dnn] overlaying the {len(_CMS)} inputs it can, in the rotated frame\n",
              flush=True)

    nano_by_kl = {_kl(d): d for d in glob.glob(os.path.join(args.nano_dir, "*kl-*")) if _kl(d) and "NanoAOD" in d}

    # Raw Delphes is one directory per kl; the merged ntuple is one file set with a kl
    # column, so drive the loop from the CMS side, which has a directory either way.
    if args.ntuple:
        # The CMS side has kl points the Delphes production has not generated yet
        # (0, 1, 5 so far). Report and skip those rather than dying part-way through.
        from delphes_pipeline.core.io import available_kl
        have = available_kl(args.ntuple)
        sources = []
        for kl in sorted(nano_by_kl):
            if any(abs(v - _kl_value(kl)) < 1e-6 for v in have):
                sources.append((kl, None))
            else:
                print(f"[overlay] skipping kl={kl}: not in the merged ntuple "
                      f"(it has {', '.join(f'{v:g}' for v in have)})")
        if not sources:
            raise SystemExit("[overlay] no kl point is in BOTH the ntuple and --nano-dir")
    else:
        sources = [(_kl(d), d) for d in sorted(glob.glob(os.path.join(args.delphes_dir, "*kl-*")))]

    for kl, d in sources:
        if kl is None or kl not in nano_by_kl:
            continue
        print(f"[kl {kl}] reconstructing features ...", flush=True)
        if args.ntuple:
            from delphes_pipeline.core.io import NtupleEvents
            dev = NtupleEvents(args.ntuple, kl=_kl_value(kl), entry_stop=args.max_events)
            print(f"[overlay] merged ntuple: {dev.n:,} events at kl={_kl_value(kl)} "
                  f"(tuning already applied at ntuplization)")
        else:
            dev = DelphesEvents(d, entry_stop=args.max_events)
        if not args.ntuple and tuning is not None:
            from delphes_pipeline.tuning.maps import RetaggedEvents
            dev = RetaggedEvents(dev, tuning, np.random.default_rng(0))
            # Say WHICH corrections actually fired. The maps file drives this, and a file
            # predating a map degrades silently: an overlay run against maps without
            # tau_response quietly falls back to the multiplicative escale and looks
            # identical to an untuned run, with nothing in the log to say so.
            print(f"[overlay] corrections applied: "
                  f"{', '.join(sorted(dev.retagged_fields)) or '(none)'}")
        sel_kw = dict(tautau_only=args.tautau_only, mtautau_min=args.mtautau_min, clean=args.clean,
                      jet_pt_min=args.jet_pt_min, jet_eta_max=args.jet_eta_max, clean_dr=args.clean_dr,
                      tau_pt_min=args.tau_pt_min, tau_eta_max=args.tau_eta_max,
                      cms_dnn=args.cms_dnn)
        nev = NanoAODEvents(nano_by_kl[kl], branches=branches, wp=wp, entry_stop=args.max_events)
        # Say WHAT was read on each side. resolve_paths recurses, so a stray dataset
        # nested under a kl directory is silently read as signal -- and it only shows up
        # once max-events is large enough to reach those files, which makes it look like
        # a physics disagreement that "appeared at full statistics".
        for lab, ev_ in (("Delphes", dev), ("CMS", nev)):
            used = getattr(ev_, "_used", None) or getattr(ev_, "paths", [])
            trees = sorted({os.path.basename(os.path.dirname(f)) for f in used})
            print(f"[overlay]   {lab:8s} {len(used)} file(s) from {len(trees)} dir(s)"
                  + (f": {', '.join(trees[:4])}" if len(trees) > 1 else ""))
            if len(trees) > 1:
                print(f"[overlay]   WARNING: {lab} input spans several directories — "
                      f"check for a stray dataset under the kl tree")
        if args.split_gen_matched:
            df, dm = features(dev, nano=False, with_match=True, **sel_kw)
            nf, nm = features(nev, nano=True, with_match=True, **sel_kw)
            _split_figure(kl, df, dm, nf, nm, args, tuning)
            continue
        df = features(dev, nano=False, **sel_kw)
        nf = features(nev, nano=True, **sel_kw)

        feats = _CMS if args.cms_dnn else _FEATURES + (_DIAG if args.diagnostics else [])
        nrow = (len(feats) + 4) // 5
        fig, axes = plt.subplots(nrow, 5, figsize=(20, 3.6 * nrow))
        for ax in axes.flat[len(feats):]:
            ax.axis("off")
        for ax, feat in zip(axes.flat, feats):
            lo, hi = _cms_range(feat) if args.cms_dnn else _RANGES[feat]
            b = np.linspace(lo, hi, 41)
            centres = 0.5 * (b[:-1] + b[1:])
            for data, lab in ((df[feat], "Delphes"), (nf[feat], "NanoAOD")):
                d = data[(data >= lo) & (data <= hi)]          # normalize over the plotted range
                if d.size:
                    h, _ = np.histogram(d, bins=b, density=True)
                    ax.step(centres, h, where="mid", lw=2, label=f"{lab} ({d.size})")
            # A near-flat density (cosThetaStar sits at ~1.0 everywhere) autoscales to a
            # sliver of y-range, which turns ordinary Poisson noise into an alarming
            # sawtooth. Anchoring at 0 puts every panel on the same footing.
            ax.set_ylim(bottom=0)
            ax.set_xlabel(feat); ax.legend(fontsize=8)
        fig.suptitle(f"$\\kappa_\\lambda$ = {kl}" + ("  (tuned)" if tuning is not None else "  (stock)")
                     + ("  · symmetric cleaning" if args.clean else "  · legacy selection"))
        out = os.path.join(args.out, f"{'cmsdnn' if args.cms_dnn else 'nsbi'}_{kl}.png")
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        print(f"[kl {kl}] -> {out}  (Delphes {df['mHH'].size}, NanoAOD {nf['mHH'].size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
