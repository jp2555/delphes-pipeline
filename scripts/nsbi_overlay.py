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


def features(ev, *, nano, tautau_only=False, mtautau_min=20.0, clean=True,
             jet_pt_min=20.0, jet_eta_max=2.4, clean_dr=0.4, with_match=False,
             tau_pt_min=20.0, tau_eta_max=2.3):
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
    }
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
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, feat in zip(axes.flat, _FEATURES):
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
        ax.set_xlabel(feat); ax.legend(fontsize=6)
    fig.suptitle(f"$\\kappa_\\lambda$ = {kl}  ·  gen-matched (solid) vs fake (dashed)"
                 + ("  · tuned" if tuning is not None else "  · stock"))
    out = os.path.join(args.out, f"split_{kl}.png")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[kl {kl}] -> {out}", flush=True)


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
    ap.add_argument("--clean", dest="clean", action="store_true", default=True,
                    help="symmetric jet cleaning on both sides (default)")
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="legacy asymmetric selection (Delphes drops τ-tagged jets, NanoAOD keeps all)")
    ap.add_argument("--jet-pt-min", type=float, default=20.0, help="common jet pT acceptance")
    ap.add_argument("--jet-eta-max", type=float, default=2.4, help="common jet |eta| acceptance")
    ap.add_argument("--clean-dr", type=float, default=0.4, help="ΔR(jet, selected τ) overlap removal")
    ap.add_argument("--tau-pt-min", type=float, default=20.0, help="common τ candidate pT acceptance")
    ap.add_argument("--tau-eta-max", type=float, default=2.3, help="common τ candidate |eta| acceptance")
    ap.add_argument("--split-gen-matched", action="store_true",
                    help="split each side into gen-matched vs fake τ pairs (diagnostic)")
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
        sel_kw = dict(tautau_only=args.tautau_only, mtautau_min=args.mtautau_min, clean=args.clean,
                      jet_pt_min=args.jet_pt_min, jet_eta_max=args.jet_eta_max, clean_dr=args.clean_dr,
                      tau_pt_min=args.tau_pt_min, tau_eta_max=args.tau_eta_max)
        nev = NanoAODEvents(nano_by_kl[kl], branches=branches, wp=wp, entry_stop=args.max_events)
        if args.split_gen_matched:
            df, dm = features(dev, nano=False, with_match=True, **sel_kw)
            nf, nm = features(nev, nano=True, with_match=True, **sel_kw)
            _split_figure(kl, df, dm, nf, nm, args, tuning)
            continue
        df = features(dev, nano=False, **sel_kw)
        nf = features(nev, nano=True, **sel_kw)

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
        fig.suptitle(f"$\\kappa_\\lambda$ = {kl}" + ("  (tuned)" if tuning is not None else "  (stock)")
                     + ("  · symmetric cleaning" if args.clean else "  · legacy selection"))
        out = os.path.join(args.out, f"nsbi_{kl}.png")
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        print(f"[kl {kl}] -> {out}  (Delphes {df['mHH'].size}, NanoAOD {nf['mHH'].size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
