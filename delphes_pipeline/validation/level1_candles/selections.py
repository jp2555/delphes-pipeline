"""Shared Level-1 candle selections (note §6.2).

The candles run on background samples, so these operate on a ``DelphesEvents``
of the candle sample (not the signal). All masks are per-event numpy booleans;
the τ-candidate helpers build the ℓτ_h / τ_hτ_h visible system for the Z→ττ
candle.
"""

from __future__ import annotations

import awkward as ak
import numpy as np


def _leading(coll: ak.Array) -> ak.Array:
    """Each event's collection sorted by descending pT."""
    return coll[ak.argsort(coll.pt, axis=1, ascending=False)]


def emu_os_mask(ev) -> np.ndarray:
    """Per-event: ≥1 electron and ≥1 muon, leading pair opposite-sign (tt̄ candle).

    The eμ requirement removes Z by construction, giving a high-purity tt̄ sample.
    """
    e, m = ev.electrons, ev.muons
    has = (ak.num(e) >= 1) & (ak.num(m) >= 1)
    e_q = ak.firsts(_leading(e).charge)
    m_q = ak.firsts(_leading(m).charge)
    os = ak.fill_none(e_q * m_q < 0, False)
    return ak.to_numpy(has & os)


def eb_insitu(ev, event_mask, *, eta_max=2.4, pt_min=20.0):
    """In-situ b-tag efficiency ε_b = 2N₂/(N₁+2N₂) (note Eq. 2).

    Over ``event_mask`` events with exactly two truth b-jets (Jet.Flavor==5) in
    acceptance, N₁/N₂ are the exactly-one-tag / two-tag counts. Returns
    ``(eb, N1, N2, bb_pt)`` where ``bb_pt`` are the truth-b pT (for the tuned-input
    comparison).
    """
    jets = ev.jets[event_mask]
    bjets = jets[(np.abs(jets.eta) <= eta_max) & (jets.flavor == 5) & (jets.pt > pt_min)]
    bb = bjets[ak.num(bjets) == 2]
    n_tag = ak.sum(bb.btag == 1, axis=1)
    N1 = int(ak.sum(n_tag == 1))
    N2 = int(ak.sum(n_tag == 2))
    denom = N1 + 2 * N2
    eb = (2.0 * N2 / denom) if denom > 0 else float("nan")
    return eb, N1, N2, ak.to_numpy(ak.flatten(bb.pt))


def bjet_veto_mask(ev) -> np.ndarray:
    """Per-event: no b-tagged jet (Z→ττ candle; decouples from b-tag tuning)."""
    return ak.to_numpy(ak.sum(ev.jets.btag == 1, axis=1) == 0)


def tau_candidates(ev) -> ak.Array:
    """Per-event τ candidates: TauTag jets (τ_h) + reco e/μ, with a kind flag.

    Fields: pt, eta, phi, mass, charge, is_tauh (1 for τ_h jets, 0 for leptons).
    Lepton charge is reliable; the τ_h-jet charge is the (approximate) jet charge.
    """
    j = ev.jets
    th = j[j.tautag == 1]
    tauh = ak.zip({"pt": th.pt, "eta": th.eta, "phi": th.phi, "mass": th.mass,
                   "charge": th.charge, "is_tauh": ak.ones_like(th.pt)})

    def lep(coll):
        return ak.zip({"pt": coll.pt, "eta": coll.eta, "phi": coll.phi,
                       "mass": ak.zeros_like(coll.pt), "charge": coll.charge,
                       "is_tauh": ak.zeros_like(coll.pt)})

    return ak.concatenate([tauh, lep(ev.electrons), lep(ev.muons)], axis=1)


def leading_visible_pair(ev, event_mask):
    """Visible m_ττ of the two leading-pT τ candidates in ``event_mask`` events.

    Returns ``(mass, n_tauh)`` numpy arrays over the selected events with ≥2
    candidates: ``mass`` is the visible di-τ invariant mass and ``n_tauh`` is how
    many of the two are hadronic (2 = τ_hτ_h, 1 = ℓτ_h, 0 = ℓℓ).
    """
    cand = tau_candidates(ev)[event_mask]
    cand = cand[ak.num(cand) >= 2]
    cand = cand[ak.argsort(cand.pt, axis=1, ascending=False)][:, :2]
    mass = _pair_mass(cand)
    n_tauh = ak.to_numpy(ak.sum(cand.is_tauh, axis=1))
    return mass, n_tauh


def _pair_mass(pair) -> np.ndarray:
    pt, eta, phi, mass = pair.pt, pair.eta, pair.phi, pair.mass
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    m2 = ak.sum(e, axis=1) ** 2 - (ak.sum(px, axis=1) ** 2 + ak.sum(py, axis=1) ** 2 + ak.sum(pz, axis=1) ** 2)
    return np.sqrt(np.maximum(ak.to_numpy(m2), 0.0))
