"""Measure tuning targets from the private NanoAOD anchor (note §3, §6.4).

The Delphes object response is tuned to match the *same* response measured on the
CMS NanoAOD. Because ``NanoAODEvents`` duck-types ``DelphesEvents``, the b-tag and
lepton targets reuse ``core.observables`` directly; the τ_h efficiency is bespoke
(NanoAOD ``GenVisTau`` matched to a ``Tau`` passing the DeepTau VSjet Medium WP);
MET is the overall resolution. Each returns a ``Profile`` used by the tuning
report as the target curve.
"""

from __future__ import annotations

from typing import Optional

import awkward as ak
import numpy as np

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.matching import nearest_target_field
from delphes_pipeline.core.nanoaod import NanoAODEvents
from delphes_pipeline.core.observables import Profile

# observables for which the NanoAOD anchor provides a target
ANCHOR_OBSERVABLES = ("btag_eff_b", "btag_eff_c", "btag_mistag_light",
                      "electron_eff", "muon_eff", "tau_eff", "met_resolution")


def anchor_profiles(config: dict, *, bins, max_events: Optional[int] = None) -> dict[str, Profile]:
    """Measure the anchor target for each observable; empty if anchor disabled."""
    ac = config.get("anchor", {})
    if not ac.get("enabled"):
        return {}
    nano = NanoAODEvents(
        ac["nanoaod_path"], branches=ac.get("branches"), wp=_resolve_wp(ac.get("wp", {})),
        entry_stop=ac.get("max_events", max_events),
    )
    out: dict[str, Profile] = {}
    for q in ("btag_eff_b", "btag_eff_c", "btag_mistag_light"):
        out[q] = obs.btag_efficiency(nano, q, bins=bins)
    for q in ("electron_eff", "muon_eff"):
        out[q] = obs.lepton_efficiency(nano, q, bins=bins)
    out["tau_eff"] = _nano_tau_eff(nano, bins)
    out["met_resolution"] = _nano_met_resolution(nano)
    # label the source for the report/plot
    for p in out.values():
        p.ylabel = (p.ylabel or "") + " (NanoAOD anchor)"
    return out


def _resolve_wp(wp: dict) -> dict:
    """Fill ``btag_medium`` from jsonpog-integration (CVMFS) when not set explicitly."""
    wp = dict(wp)
    if wp.get("btag_medium") is None and wp.get("btag_correctionlib"):
        from . import correctionlib_wp
        wp["btag_medium"] = correctionlib_wp.resolve_btag_wp(wp["btag_correctionlib"])
        print(f"[tuning] resolved btag_medium = {wp['btag_medium']:.4f} from jsonpog-integration")
    return wp


def _nano_tau_eff(nano: NanoAODEvents, bins, *, dr=0.4, eta_max=2.5, pt_min=20.0) -> Profile:
    """τ_h efficiency on NanoAOD: GenVisTau matched to a DeepTau-Medium Tau."""
    gvt = nano.genvistau
    acc = gvt[(np.abs(gvt.eta) <= eta_max) & (gvt.pt > pt_min)]
    matched, vsjet = nearest_target_field(acc, nano.taus, dr, "vsjet")
    passed = matched & (np.nan_to_num(np.asarray(vsjet), nan=-1.0) >= nano.deeptau_medium())
    prof = obs.binned_efficiency(ak.to_numpy(ak.flatten(acc.pt)), passed, bins, quantity="tau_eff", x="pt")
    prof.xlabel, prof.ylabel = "tau pT [GeV]", "tau_eff"
    return prof


def _nano_met_resolution(nano: NanoAODEvents) -> Profile:
    """Overall MET resolution as a single-bin profile (ΣE_T definitions differ)."""
    dx, dy, _ = obs.met_residuals(nano)
    res = float(np.sqrt(0.5 * (np.var(dx) + np.var(dy)))) if dx.size else float("nan")
    err = res / np.sqrt(2.0 * max(dx.size, 1))
    return Profile("met_resolution", "sumet", np.array([0.0]), np.array([res]),
                   np.array([err]), np.array([dx.size], dtype=int), kind="resolution",
                   xlabel="(overall)", ylabel="MET resolution [GeV]")
