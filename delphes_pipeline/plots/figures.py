"""Validation + signal-baseline figures.

Object-level figures justify the card patches; signal-baseline figures argue the
CMS bbtautau baseline is reproducible. Data-driven figures take a ``DelphesEvents``
(or several, for the kappa_lambda overlay); ``lepton_eff_floor_curves`` is pure
card formula and needs no data. Each returns the output PNG path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.core.plotting import cms_style, hist_overlay
from . import quantities as q


# --------------------------------------------------------------------------- #
# Patch-justification figures (data-driven)
# --------------------------------------------------------------------------- #
def jet_pt_spectrum(ev: DelphesEvents, outdir: Path) -> str:
    """AK4 jet pT spectrum with the 15 GeV generation floor and 20 GeV cut (PATCH-2)."""
    return hist_overlay(
        [("AK4 jets", q.jet_pt(ev), None)],
        bins=np.linspace(0, 150, 46), outpath=outdir / "jet_pt_spectrum.png",
        xlabel="jet pT [GeV]", ylabel="jets (norm.)", logy=True,
        vlines=[(15.0, "gen floor (PATCH-2)"), (20.0, "analysis cut")],
        title="AK4 jet pT: 20 GeV cut sits on a populated spectrum",
    )


def tauh_pt_spectrum(ev: DelphesEvents, outdir: Path) -> str:
    """tau-candidate (TauTag jet) pT near the 20 GeV cut (PATCH-2)."""
    return hist_overlay(
        [("tau-candidate jets", q.tauh_pt(ev), None)],
        bins=np.linspace(0, 120, 41), outpath=outdir / "tauh_pt_spectrum.png",
        xlabel="tau-candidate pT [GeV]", ylabel="candidates (norm.)",
        vlines=[(20.0, "tau_h cut")],
        title="tau_h candidate pT is sane at 20-25 GeV with the 15 GeV jet floor",
    )


def jet_eta_spectrum(ev: DelphesEvents, outdir: Path) -> str:
    """AK4 jet |eta| with the b-tag (2.4) acceptance marked (PATCH-1)."""
    return hist_overlay(
        [("AK4 jets", q.jet_eta(ev), None), ("b jets (flavor 5)", _bjet_eta(ev), None)],
        bins=np.linspace(-5, 5, 51), outpath=outdir / "jet_eta_spectrum.png",
        xlabel="jet eta", ylabel="jets (norm.)",
        vlines=[(-2.4, "b-tag |eta|<2.4"), (2.4, "")],
        title="AK4 (R=0.4) jet eta; b-tagging restricted to |eta|<2.4",
    )


def mbb_peak_figure(ev: DelphesEvents, outdir: Path) -> str:
    """Visible AK4 di-b-jet mass peak (PATCH-1 correctness)."""
    return hist_overlay(
        [("two highest-BTag jets", q.mbb_values(ev), None)],
        bins=np.linspace(0, 300, 61), outpath=outdir / "mbb_peak.png",
        xlabel="m_bb [GeV]", ylabel="events (norm.)",
        vlines=[(125.0, "m_H")],
        title="Visible m_bb (AK4 R=0.4, no b-energy regression)",
    )


def lepton_pt_spectra(ev: DelphesEvents, outdir: Path) -> str:
    """Reco e/mu pT reaching below the veto thresholds to the patched floors (PATCH-3/4)."""
    return hist_overlay(
        [("electrons", q.lepton_pt(ev, "electron"), None),
         ("muons", q.lepton_pt(ev, "muon"), None)],
        bins=np.linspace(0, 60, 61), outpath=outdir / "lepton_pt_spectra.png",
        xlabel="lepton pT [GeV]", ylabel="leptons (norm.)", logy=True,
        vlines=[(5.0, "mu floor 5"), (7.0, "e floor 7"), (6.0, "mu veto"), (10.0, "e veto")],
        title="Reco leptons populate below the veto thresholds (vetoes emulatable)",
    )


# --------------------------------------------------------------------------- #
# Patch-justification figure (pure card formula, no data)
# --------------------------------------------------------------------------- #
def lepton_eff_floor_curves(outdir: Path) -> str:
    """Stock-vs-patched lepton ID efficiency floors with the veto thresholds (PATCH-3/4)."""
    cms_style()
    pt = np.linspace(0, 30, 600)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for ax, (name, stock_floor, patch_floor, veto, color) in zip(
        axes,
        [("electron", 10.0, 7.0, 10.0, "C0"), ("muon", 10.0, 5.0, 6.0, "C3")],
    ):
        ax.plot(pt, 0.95 * (pt > stock_floor), color="0.6", lw=2, label=f"stock (floor {stock_floor:.0f} GeV)")
        ax.plot(pt, 0.95 * (pt > patch_floor), color=color, lw=2, label=f"patched (floor {patch_floor:.0f} GeV)")
        ax.axvline(veto, ls="--", color="k", lw=1.2)
        ax.text(veto, 0.5, f" {name} veto @ {veto:.0f}", rotation=90, va="center", fontsize=8)
        ax.set_title(f"{name} ID efficiency (barrel)")
        ax.set_xlabel("pT [GeV]")
        ax.set_ylabel("ID efficiency")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.suptitle("PATCH-3/4: floors lowered so the third-lepton veto is emulatable", fontsize=11)
    out = Path(outdir) / "lepton_eff_floor_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------- #
# Signal-baseline figures (data-driven)
# --------------------------------------------------------------------------- #
def mhh_klambda_overlay(samples: "Mapping[str, DelphesEvents]", outdir: Path) -> str:
    """Gen-level m_HH overlaid across the kappa_lambda points (the baseline plot)."""
    series = [(label, q.gen_mhh(ev), None) for label, ev in samples.items()]
    return hist_overlay(
        series, bins=np.linspace(200, 1000, 41), outpath=outdir / "mhh_klambda.png",
        xlabel="gen m_HH [GeV]", ylabel="events (norm.)",
        title="Gen m_HH shape vs kappa_lambda (threshold enhancement / SM dip / high-mass tail)",
    )


def mtautau_figure(ev: DelphesEvents, outdir: Path) -> str:
    """Visible di-tau mass peak."""
    return hist_overlay(
        [("visible di-tau", q.mtautau_visible(ev), None)],
        bins=np.linspace(0, 250, 51), outpath=outdir / "mtautau_visible.png",
        xlabel="visible m_tautau [GeV]", ylabel="events (norm.)",
        vlines=[(125.0, "m_H")],
        title="Visible di-tau mass (below m_H; neutrinos missing)",
    )


def multiplicity_figure(ev: DelphesEvents, outdir: Path) -> str:
    """Object multiplicity summary (n jets / b-tag / tau-cand / leptons)."""
    m = q.multiplicities(ev)
    bins = np.arange(-0.5, 10.5, 1.0)
    return hist_overlay(
        [(k.replace("n_", ""), v, None) for k, v in m.items()],
        bins=bins, outpath=outdir / "multiplicities.png",
        xlabel="object count / event", ylabel="events (norm.)",
        title="Object multiplicities",
    )


def _bjet_eta(ev: DelphesEvents) -> np.ndarray:
    import awkward as ak
    j = ev.jets
    return ak.to_numpy(ak.flatten(j.eta[j.flavor == 5]))
