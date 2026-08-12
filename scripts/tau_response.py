"""Overlay the τ_h energy RESPONSE distribution — Delphes vs CMS — not just its median.

    pixi run python scripts/tau_response.py --config config.v1.yml \
        --delphes-root '/ceph/jpan/gen-delphes/*kl-1p00*Delphes_v1' \
        --nano-path   '/ceph/jpan/cms_nanoaod_2024_hh2b2tau/*kl-1p00*NanoAODv15*' \
        --out plots/tau_response --max-events 20000

``tau_escale`` is a per-pT-bin MEDIAN ratio applied multiplicatively, so it can align
medians and nothing else. If the Delphes response is one-sidedly broader than CMS at
matched median, that excess is unreachable by any multiplicative map: a scale factor
cannot depopulate a kinematically forbidden region. The lower row here — each side
divided by its OWN per-bin median — is therefore the decisive panel: it removes the scale
difference by construction and shows only what the map cannot fix.

Two knobs, because the derived map's two sides do NOT currently use the same convention:

  --gate {gen,reco}   which object the pT/|eta| acceptance is applied to. The anchor gates
                      on the GEN visible τ; the Delphes side gates on the RECO jet. Gating
                      on reco drops gen τ whose reco object fluctuated low, so the two
                      populations differ. Default ``gen`` = symmetric.
  --require-id        condition the reco object on the τ ID (Delphes TauTag / DeepTau
                      Medium). The anchor ALWAYS does this; the Delphes side never does.
                      This asymmetry is not cosmetic: DeepTau is isolation-based and
                      preferentially rejects contaminated candidates, whereas the Delphes
                      tag is a stochastic draw uncorrelated with contamination. Compare
                      with and without to size it.
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
from delphes_pipeline.core.matching import nearest_target_field
from delphes_pipeline.core.nanoaod import NanoAODEvents
from delphes_pipeline.validation.run_validation import load_config

# wide enough to hold the whole tail: the point of the study is what sits above ~1.5
_BINS = [(20, 30), (30, 50), (50, 100), (100, 300)]
_TAILQ = [0.5, 0.75, 0.90, 0.95, 0.99]


def _response(gen_obj, reco_obj, *, gate, dr=0.4, eta_max=2.5, pt_min=20.0):
    """(gen_pt, reco/gen) per matched τ, with the acceptance applied to one side only.

    ``gate="gen"`` probes from the gen visible τ and asks for a reco object — the
    anchor's convention. ``gate="reco"`` probes from the reco object and asks for a gen
    τ — the Delphes side's current convention, which silently drops gen τ whose reco
    partner fell below the pT floor.
    """
    if gate == "gen":
        acc = gen_obj[(np.abs(gen_obj.eta) <= eta_max) & (gen_obj.pt > pt_min)]
        matched, reco_pt = nearest_target_field(acc, reco_obj, dr, "pt")
        gen_pt = ak.to_numpy(ak.flatten(acc.pt))
        reco_pt = np.nan_to_num(reco_pt, nan=0.0)
    else:
        acc = reco_obj[(np.abs(reco_obj.eta) <= eta_max) & (reco_obj.pt > pt_min)]
        matched, gen_pt = nearest_target_field(acc, gen_obj, dr, "pt")
        reco_pt = ak.to_numpy(ak.flatten(acc.pt))
        gen_pt = np.nan_to_num(gen_pt, nan=0.0)
    ok = matched & (gen_pt > 0) & (reco_pt > 0)
    return gen_pt[ok], reco_pt[ok] / gen_pt[ok]


def delphes_response(ev, *, gate, require_id):
    """A Delphes τ_h is a jet; ``require_id`` restricts to the (stochastic) TauTag bit."""
    jets = ev.jets
    if require_id:
        jets = jets[jets.tautag == 1]
    return _response(obs.gen_visible_taus(ev.gen, dr=0.4), jets, gate=gate)


def nano_response(ev, *, gate, require_id):
    taus = ev.taus
    if require_id:
        taus = taus[taus.vsjet >= ev.deeptau_medium()]
    return _response(ev.genvistau, taus, gate=gate)


def _tail_table(dg, dr_, ng, nr):
    """Quantiles of response/median per pT bin — the tail, with the scale divided out."""
    print(f"\n{'pT bin':>10s}  {'side':>8s}  {'n':>6s}  {'median':>7s}  "
          + "  ".join(f"q{int(q*100):02d}/med" for q in _TAILQ[1:]))
    for lo, hi in _BINS:
        for name, g, r in (("Delphes", dg, dr_), ("CMS", ng, nr)):
            m = (g >= lo) & (g < hi)
            if m.sum() < 20:
                print(f"{f'{lo}-{hi}':>10s}  {name:>8s}  {int(m.sum()):>6d}   (too few)")
                continue
            v = r[m]
            med = float(np.median(v))
            qs = [np.quantile(v, q) / med for q in _TAILQ[1:]]
            print(f"{f'{lo}-{hi}':>10s}  {name:>8s}  {int(m.sum()):>6d}  {med:7.3f}  "
                  + "  ".join(f"{q:8.3f}" for q in qs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--delphes-root", required=True)
    ap.add_argument("--nano-path", required=True)
    ap.add_argument("--out", default="plots/tau_response")
    ap.add_argument("--max-events", type=int, default=20000)
    ap.add_argument("--gate", choices=("gen", "reco"), default="gen",
                    help="apply the pT/|eta| acceptance to the gen or the reco object")
    ap.add_argument("--require-id", action="store_true",
                    help="condition the reco object on the tau ID on BOTH sides")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ac = cfg.get("anchor", {})
    dev = DelphesEvents(args.delphes_root, treename=cfg.get("input", {}).get("treename", "Delphes"),
                        entry_stop=args.max_events)
    nev = NanoAODEvents(args.nano_path, branches=ac.get("branches"),
                        wp={"deeptau_vsjet_medium": ac.get("wp", {}).get("deeptau_vsjet_medium", 5)},
                        entry_stop=args.max_events)
    kw = dict(gate=args.gate, require_id=args.require_id)
    dg, dr_ = delphes_response(dev, **kw)
    ng, nr = nano_response(nev, **kw)
    print(f"[response] gate={args.gate} require_id={args.require_id}  "
          f"Delphes {dg.size} tau, CMS {ng.size} tau")
    _tail_table(dg, dr_, ng, nr)

    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(2, len(_BINS), figsize=(4.6 * len(_BINS), 8))
    for k, (lo, hi) in enumerate(_BINS):
        md, mn = (dg >= lo) & (dg < hi), (ng >= lo) & (ng < hi)
        vd, vn = dr_[md], nr[mn]
        # top: as measured -- shows the SCALE difference the map is built to remove
        ax = axes[0, k]
        for v, name in ((vd, "Delphes"), (vn, "CMS")):
            if v.size:
                ax.hist(v, bins=60, range=(0, 3), histtype="step", density=True,
                        label=f"{name} ({v.size})")
        ax.set_xlabel("reco / gen-visible pT"), ax.set_title(f"gen pT {lo}-{hi} GeV")
        ax.legend(fontsize=7)
        # bottom: each divided by its OWN median -- the SHAPE, which the map cannot touch
        ax = axes[1, k]
        for v, lab in ((vd, "Delphes"), (vn, "CMS")):
            if v.size:
                ax.hist(v / np.median(v), bins=60, range=(0, 3), histtype="step",
                        density=True, label=lab)
        ax.set_xlabel("response / its own median"), ax.set_yscale("log")
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("normalised")
    axes[1, 0].set_ylabel("normalised (log)")
    fig.suptitle(f"tau_h energy response  ·  gate={args.gate}"
                 f"{'  ·  tau-ID required' if args.require_id else '  ·  no tau ID'}"
                 "   —   bottom row is the shape the escale map CANNOT change")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"tau_response.{ext}"), dpi=130)
    print(f"[response] wrote {args.out}/tau_response.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
