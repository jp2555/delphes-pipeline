"""Raw kinematic quantities for the validation/baseline figures.

Thin extractors returning flat numpy arrays from a ``DelphesEvents``. Physics
quantities shared with the gate (m_bb, energy response) come from
``core.observables``; this module adds the spectra and the gen-level m_HH used
only by the plots.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.core.observables import _pair_mass, mbb_values

__all__ = ["jet_pt", "jet_eta", "bjet_pt", "tauh_pt", "lepton_pt",
           "mtautau_visible", "gen_mhh", "multiplicities", "mbb_values"]

_HIGGS_PID = 25


def jet_pt(events: DelphesEvents) -> np.ndarray:
    return ak.to_numpy(ak.flatten(events.jets.pt))


def jet_eta(events: DelphesEvents) -> np.ndarray:
    return ak.to_numpy(ak.flatten(events.jets.eta))


def bjet_pt(events: DelphesEvents) -> np.ndarray:
    j = events.jets
    return ak.to_numpy(ak.flatten(j.pt[j.flavor == 5]))


def tauh_pt(events: DelphesEvents) -> np.ndarray:
    """pT of reco tau-candidate jets (Jet.TauTag == 1)."""
    j = events.jets
    return ak.to_numpy(ak.flatten(j.pt[j.tautag == 1]))


def lepton_pt(events: DelphesEvents, flavor: str) -> np.ndarray:
    coll = events.electrons if flavor == "electron" else events.muons
    return ak.to_numpy(ak.flatten(coll.pt))


def mtautau_visible(events: DelphesEvents) -> np.ndarray:
    """Visible di-tau mass: invariant mass of the two leading tau candidates.

    Tau candidates = TauTag jets + reco electrons + muons; the two highest-pT are
    combined. Channel-inclusive and visible-only, so the peak sits below m_H.
    """
    j = events.jets
    taus = j[j.tautag == 1]
    cand = _concat_candidates(taus, events.electrons, events.muons)
    lead = cand[ak.num(cand) >= 2]
    lead = lead[ak.argsort(lead.pt, axis=1, ascending=False, stable=True)][:, :2]
    return _pair_mass(lead)


def gen_mhh(events: DelphesEvents) -> np.ndarray:
    """Gen-level m_HH from the two leading-pT gen Higgs bosons (|pdgId| == 25).

    Heuristic: prefer hard/intermediate copies (status >= 20) when a status field
    is present, then take the two leading-pT Higgs. On real Powheg+Pythia samples
    verify this selects the two physical Higgs rather than two copies of one.
    """
    gen = events.gen
    h = gen[np.abs(gen.pid) == _HIGGS_PID]
    if "status" in h.fields:
        hard = h[h.status >= 20]
        h = ak.where(ak.num(hard) >= 2, hard, h)
    h = h[ak.num(h) >= 2]
    h = h[ak.argsort(h.pt, axis=1, ascending=False, stable=True)][:, :2]
    return _pair_mass(h)


def multiplicities(events: DelphesEvents) -> dict[str, np.ndarray]:
    """Per-event object counts (n jets, b-tag jets, tau-candidates, leptons)."""
    j = events.jets
    return {
        "n_jets": ak.to_numpy(ak.num(j)),
        "n_btag": ak.to_numpy(ak.sum(j.btag == 1, axis=1)),
        "n_tauh": ak.to_numpy(ak.sum(j.tautag == 1, axis=1)),
        "n_leptons": ak.to_numpy(ak.num(events.electrons) + ak.num(events.muons)),
    }


def _concat_candidates(taus, electrons, muons) -> ak.Array:
    """Per-event concatenation of (pt, eta, phi, mass) over tau-jets + leptons."""
    def rec(coll, massless):
        mass = ak.zeros_like(coll.pt) if massless else coll.mass
        return ak.zip({"pt": coll.pt, "eta": coll.eta, "phi": coll.phi, "mass": mass})

    return ak.concatenate(
        [rec(taus, massless=False), rec(electrons, massless=True), rec(muons, massless=True)],
        axis=1,
    )
