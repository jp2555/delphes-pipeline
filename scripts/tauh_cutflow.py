"""Where does the Delphes τ_hτ_h yield go? A step-by-step cutflow vs CMS NanoAOD.

    pixi run python scripts/tauh_cutflow.py --config config.v1.yml \
        --delphes-dir /ceph/jpan/gen-delphes \
        --nano-dir    /ceph/jpan/cms_nanoaod_2024_hh2b2tau \
        --max-events 20000 [--kl 1p00] [--no-tuned] [--out plots/cutflow]

The NSBI overlay selects ~5x fewer τ_hτ_h events on Delphes than on CMS from the same
number of generated events. That deficit caps the NSBI training statistics, so it
matters *which* step loses them. Each side is walked through the same five stages and
the per-step (conditional) efficiency is compared:

  1. gen  — >=2 VISIBLE hadronic τ in acceptance. NanoAOD has ``GenVisTau``; the Delphes
     analogue is a ``GenJet`` matched to a gen τ (GenJets run after ``NeutrinoFilter``,
     so they are already neutrino-filtered, i.e. visible). Binning the Delphes side by
     the *full* gen-τ pT instead would compare different quantities — a hadronic τ shows
     only ~65% of its pT.
  2. reco — >=2 reconstructed τ_h candidate OBJECTS matched to those visible τ
     (Delphes: an accepted jet; CMS: a ``Tau``). Isolates "did the object get made".
  3. id   — >=2 of them pass the τ_h ID (Delphes: the re-tagged ``TauTag``; CMS: DeepTau
     VSjet Medium). Isolates the identification efficiency.
  4. jets — >=2 jets survive the symmetric cleaning (ΔR from the selected τ pair).
  5. mtt  — the FastMTT di-τ mass is finite and above ``--mtautau-min``.

A step whose *conditional* efficiency differs between the two columns is where the
deficit is created; steps that agree are not the problem.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import awkward as ak
import numpy as np

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.core.matching import matched_to_any
from delphes_pipeline.core.nanoaod import NanoAODEvents
from delphes_pipeline.validation.run_validation import load_config

_KL = re.compile(r"kl-(m?\d+p\d+)")
_STEPS = ["events read", ">=2 gen visible tau_h", ">=2 reco tau_h object",
          ">=2 pass tau_h ID", ">=2 cleaned jets", "FastMTT m_tautau ok"]


def _kl(path):
    m = _KL.search(os.path.basename(os.path.normpath(path)))
    return m.group(1) if m else None


def _acc(coll, pt_min, eta_max):
    return coll[(coll.pt > pt_min) & (np.abs(coll.eta) <= eta_max)]


def _gen_dr(vis) -> np.ndarray:
    """ΔR between the two leading-pT visible gen τ (NaN where there are fewer than 2)."""
    v = vis[ak.argsort(vis.pt, axis=1, ascending=False, stable=True)]
    has2 = ak.to_numpy(ak.num(v) >= 2)
    out = np.full(has2.shape, np.nan)
    if has2.any():
        p = v[has2][:, :2]
        eta = ak.to_numpy(p.eta)
        phi = ak.to_numpy(p.phi)
        dphi = np.abs((phi[:, 0] - phi[:, 1] + np.pi) % (2 * np.pi) - np.pi)
        out[has2] = np.hypot(eta[:, 0] - eta[:, 1], dphi)
    return out


def _cutflow(ev, *, nano, pt_min=20.0, eta_max=2.3, dr=0.4, jet_pt_min=20.0,
             jet_eta_max=2.4, clean_dr=0.4, mtautau_min=20.0, detail=False):
    """Per-step surviving-event counts for one sample. Returns a list of ints.

    With ``detail`` also returns ``{gen_dr, s1, s5}`` — the generator-level ΔR between
    the two visible hadronic τ and the stage-1 / stage-5 masks. That is what separates a
    *reconstruction* difference (the two sides start from the same gen ΔR spectrum but
    select it with different efficiency) from a *sample* difference (the gen spectra
    themselves differ, in which case no amount of detector tuning will close it).
    """
    if nano:
        vis = _acc(ev.genvistau, pt_min, eta_max)
        cand_all = _acc(ev.taus, pt_min, eta_max)
        passes_id = lambda c: c[c.vsjet >= ev.deeptau_medium()]
    else:
        # The ν-subtracted hadronic gen τ — the true GenVisTau analogue. A GenJet matched
        # to a τ is NOT that: it is an R=0.4 cluster carrying the underlying event, ~17%
        # harder than the visible τ at low pT, so cutting pT>20 on it let ~17% more events
        # through than the CMS side. Same GenJet-vs-GenVisTau reference error that the τ
        # energy scale had. Verified against CMS GenVisTau by scripts/gen_tau_check.py:
        # this construction reproduces CMS's stage-1 rate to ~1%.
        vis = _acc(obs.gen_visible_taus(ev.gen, dr=dr), pt_min, eta_max)
        cand_all = _acc(ev.jets, pt_min, eta_max)
        passes_id = lambda c: c[c.tautag == 1]

    n_events = int(ak.num(vis, axis=0))
    s1 = ak.to_numpy(ak.num(vis) >= 2)

    reco = cand_all[matched_to_any(cand_all, vis, dr)]      # object matched to a visible τ
    s2 = s1 & ak.to_numpy(ak.num(reco) >= 2)

    tau_id = passes_id(reco)
    s3 = s2 & ak.to_numpy(ak.num(tau_id) >= 2)

    # step 4: the symmetric cleaning of the overlay — pick the τ pair, then the jets
    pair = tau_id[ak.argsort(tau_id.pt, axis=1, ascending=False, stable=True)][:, :2]
    jets = _acc(ev.jets, jet_pt_min, jet_eta_max)
    keep = s3
    jets_k, pair_k = jets[keep], pair[keep]
    clean = jets_k[~matched_to_any(jets_k, pair_k, clean_dr)]
    s4 = keep.copy()
    s4[keep] = ak.to_numpy(ak.num(clean) >= 2)

    # step 5: FastMTT on the surviving events
    from delphes_pipeline.extensions.mtautau import _leg, fastmtt_mass
    idx = np.flatnonzero(s4)
    s5 = s4.copy()
    if idx.size:
        p2 = pair[s4]
        met = ev.met[s4]
        mx = ak.to_numpy(ak.fill_none(met.met * np.cos(met.phi), np.nan))
        my = ak.to_numpy(ak.fill_none(met.met * np.sin(met.phi), np.nan))
        legs = []
        for i in (0, 1):
            c = p2[:, i]
            legs.append(_leg(ak.zip({"pt": c.pt, "eta": c.eta, "phi": c.phi,
                                     "mass": c.mass, "is_tauh": ak.ones_like(c.pt)})))
        m = fastmtt_mass(legs[0], legs[1], mx, my)
        s5[idx] = np.isfinite(m) & (m > mtautau_min)

    counts = [n_events, int(s1.sum()), int(s2.sum()), int(s3.sum()), int(s4.sum()), int(s5.sum())]
    if detail:
        return counts, {"gen_dr": _gen_dr(vis), "s1": s1, "s5": s5}
    return counts


def _gen_dr_figure(kl, dd, dn, outdir):
    """Gen ΔR_ττ spectrum and the selection efficiency against it, for both sides.

    Panel 1 asks whether the two samples *start* from the same distribution; panel 2 asks
    whether they *select* it the same way. Only panel 2 is a detector statement.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    b = np.linspace(0.0, 5.0, 26)
    ctr = 0.5 * (b[:-1] + b[1:])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for d, lab, col in ((dd, "Delphes", "tab:blue"), (dn, "CMS", "tab:orange")):
        gen_dr, s1, s5 = d["gen_dr"], d["s1"], d["s5"]
        den_v = gen_dr[s1 & np.isfinite(gen_dr)]
        num_v = gen_dr[s5 & np.isfinite(gen_dr)]
        if den_v.size:
            h, _ = np.histogram(den_v, bins=b, density=True)
            ax1.step(ctr, h, where="mid", lw=2, color=col, label=f"{lab} ({den_v.size})")
        den, _ = np.histogram(den_v, bins=b)
        num, _ = np.histogram(num_v, bins=b)
        ok = den > 0
        eff = np.divide(num, np.maximum(den, 1), dtype=float)
        err = np.sqrt(np.maximum(eff * (1 - eff), 0) / np.maximum(den, 1))
        ax2.errorbar(ctr[ok], eff[ok], yerr=err[ok], fmt="o-", ms=3, lw=1.5,
                     color=col, label=lab)
    ax1.set_xlabel(r"gen $\Delta R_{\tau\tau}$  (>=2 visible gen $\tau_h$)")
    ax1.set_ylabel("normalised"); ax1.legend(fontsize=9)
    ax1.set_title("do the samples START the same?")
    ax2.set_xlabel(r"gen $\Delta R_{\tau\tau}$")
    ax2.set_ylabel("selection efficiency (stage 5 / stage 1)"); ax2.legend(fontsize=9)
    ax2.set_title("do they SELECT it the same?")
    fig.suptitle(f"$\\kappa_\\lambda$ = {kl}")
    out = os.path.join(outdir, f"gen_dr_{kl}.png")
    fig.tight_layout(); fig.savefig(out, dpi=115); plt.close(fig)
    print(f"[kl {kl}] -> {out}", flush=True)


def _table(d, n) -> str:
    """Markdown cutflow table with per-step and cumulative efficiencies."""
    out = ["| step | Delphes N | ε_step | ε_tot | CMS N | ε_step | ε_tot | Δε_step |",
           "|---|---|---|---|---|---|---|---|"]
    for i, name in enumerate(_STEPS):
        ed = d[i] / d[i - 1] if i and d[i - 1] else float("nan")
        en = n[i] / n[i - 1] if i and n[i - 1] else float("nan")
        td = d[i] / d[0] if d[0] else float("nan")
        tn = n[i] / n[0] if n[0] else float("nan")
        ratio = (ed / en) if i and en else float("nan")
        step_d = "—" if not i else f"{ed:.3f}"
        step_n = "—" if not i else f"{en:.3f}"
        flag = ""
        if i and np.isfinite(ratio):
            flag = f"{ratio:.2f}" + ("  ⟵" if ratio < 0.8 or ratio > 1.25 else "")
        out.append(f"| {i} {name} | {d[i]} | {step_d} | {td:.4f} | "
                   f"{n[i]} | {step_n} | {tn:.4f} | {flag} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="τ_hτ_h cutflow: Delphes vs CMS NanoAOD")
    ap.add_argument("--config", required=True)
    ap.add_argument("--delphes-dir", required=True)
    ap.add_argument("--nano-dir", required=True)
    ap.add_argument("--kl", default=None, help="only this κ_λ tag (e.g. 1p00); default all")
    ap.add_argument("--max-events", type=int, default=20000)
    ap.add_argument("--tuned", dest="tuned", action="store_true", default=True)
    ap.add_argument("--no-tuned", dest="tuned", action="store_false")
    ap.add_argument("--mtautau-min", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="write the tables to this markdown file")
    ap.add_argument("--gen-dr", default=None, metavar="DIR",
                    help="also plot the gen ΔR_ττ spectrum and the selection efficiency "
                         "vs gen ΔR_ττ into DIR (the reco-vs-sample discriminator)")
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
        print(f"[cutflow] applying tuning maps from {maps_path}")

    nano_by_kl = {_kl(d): d for d in glob.glob(os.path.join(args.nano_dir, "*kl-*"))
                  if _kl(d) and "NanoAOD" in d}
    chunks = []
    for d in sorted(glob.glob(os.path.join(args.delphes_dir, "*kl-*"))):
        kl = _kl(d)
        if kl is None or kl not in nano_by_kl or (args.kl and kl != args.kl):
            continue
        print(f"[kl {kl}] walking the cutflow ...", flush=True)
        dev = DelphesEvents(d, entry_stop=args.max_events)
        if tuning is not None:
            from delphes_pipeline.tuning.maps import RetaggedEvents
            dev = RetaggedEvents(dev, tuning, np.random.default_rng(0))
            print(f"[cutflow] corrections applied: "
                  f"{', '.join(sorted(dev.retagged_fields)) or '(none)'}")
        nev = NanoAODEvents(nano_by_kl[kl], branches=branches, wp=wp,
                            entry_stop=args.max_events)
        if args.gen_dr:
            cd, dd = _cutflow(dev, nano=False, mtautau_min=args.mtautau_min, detail=True)
            cn, dn = _cutflow(nev, nano=True, mtautau_min=args.mtautau_min, detail=True)
            _gen_dr_figure(kl, dd, dn, args.gen_dr)
        else:
            cd = _cutflow(dev, nano=False, mtautau_min=args.mtautau_min)
            cn = _cutflow(nev, nano=True, mtautau_min=args.mtautau_min)
        head = (f"\n### κ_λ = {kl}  ({'tuned' if tuning is not None else 'stock'})\n\n"
                f"final yield ratio CMS/Delphes = {cn[-1] / max(cd[-1], 1):.2f}\n")
        body = _table(cd, cn)
        print(head + body, flush=True)
        chunks.append(head + body)

    if args.out and chunks:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write("# τ_hτ_h cutflow — Delphes vs CMS NanoAOD\n"
                     "\n`Δε_step` is the Delphes/CMS ratio of the per-step efficiency; "
                     "a value away from 1 (flagged ⟵) is where the deficit is made.\n"
                     + "\n".join(chunks) + "\n")
        print(f"[cutflow] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
