"""Synthetic NanoAOD-like fixture for the tuning-anchor tests.

Writes a flat ``Events`` tree with the NanoAOD branch names the anchor reader
expects, injecting known b-tag/tau/lepton efficiencies so the anchor measurement
can be checked against ground truth. The b-tag discriminant encodes the injected
efficiency relative to a 0.5 working point; DeepTau VSjet uses the bitmask
convention (Medium = 16).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import awkward as ak
import numpy as np
import uproot

Eff = Callable[[float, float], float]

DEF_BTAG_EFF: Eff = lambda pt, eta: 0.70
DEF_CTAG_EFF: Eff = lambda pt, eta: 0.15
DEF_LTAG_MISTAG: Eff = lambda pt, eta: 0.02
DEF_TAU_EFF: Eff = lambda pt, eta: 0.60
DEF_TAU_MISTAG: Eff = lambda pt, eta: 0.03
DEF_LEP_EFF: Eff = lambda pt, eta: 0.90

BTAG_WP = 0.5            # the anchor reader thresholds the discriminant here
DEEPTAU_MEDIUM = 5       # DeepTau v2p5 VSjet Medium WP level (NanoAODv15: 0..8), reader uses >=


@dataclass
class NanoTruth:
    n_events: int
    btag_eff: Eff
    ctag_eff: Eff
    ltag_mistag: Eff
    tau_eff: Eff
    lep_eff: Eff
    tau_mistag: Eff = DEF_TAU_MISTAG
    btag_wp: float = BTAG_WP
    deeptau_medium: int = DEEPTAU_MEDIUM


def make_nano_fixture(path: str, *, n_events: int = 4000, seed: int = 0,
                      met_resolution_gev: float = 18.0,
                      btag_eff=DEF_BTAG_EFF, ctag_eff=DEF_CTAG_EFF, ltag_mistag=DEF_LTAG_MISTAG,
                      tau_eff=DEF_TAU_EFF, tau_mistag=DEF_TAU_MISTAG, lep_eff=DEF_LEP_EFF) -> NanoTruth:
    rng = np.random.default_rng(seed)
    cols: dict[str, list] = {k: [] for k in (
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_hadronFlavour", "Jet_btagUParTAK4B",
        "Tau_pt", "Tau_eta", "Tau_phi", "Tau_mass", "Tau_idDeepTau2018v2p5VSjet", "Tau_genPartFlav",
        "GenVisTau_pt", "GenVisTau_eta", "GenVisTau_phi", "GenVisTau_mass",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_charge",
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge",
        "GenPart_pdgId", "GenPart_status", "GenPart_pt", "GenPart_eta", "GenPart_phi",
        "GenPart_mass", "GenPart_genPartIdxMother")}
    scal: dict[str, list] = {k: [] for k in ("PuppiMET_pt", "PuppiMET_phi", "GenMET_pt", "GenMET_phi", "genWeight")}

    def disc(eff):  # encode efficiency relative to the 0.5 WP
        return float(rng.uniform(BTAG_WP, 1.0) if rng.random() < eff else rng.uniform(0.0, BTAG_WP))

    for _ in range(n_events):
        ev = {k: [] for k in cols}
        njet = int(rng.integers(2, 6))
        for _j in range(njet):
            pt = float(rng.exponential(40) + 20); eta = float(rng.uniform(-2.5, 2.5)); phi = float(rng.uniform(-math.pi, math.pi))
            fl = int(rng.choice([5, 4, 0], p=[0.3, 0.2, 0.5]))
            e = btag_eff(pt, eta) if fl == 5 else ctag_eff(pt, eta) if fl == 4 else ltag_mistag(pt, eta)
            _push(ev, "Jet", pt, eta, phi, 8.0); ev["Jet_hadronFlavour"].append(fl); ev["Jet_btagUParTAK4B"].append(disc(e))
            if rng.random() < tau_mistag(pt, eta):  # jet fakes a τ_h: a Medium Tau at the jet
                _push(ev, "Tau", pt, eta, phi, 1.0)
                ev["Tau_idDeepTau2018v2p5VSjet"].append(6); ev["Tau_genPartFlav"].append(0)

        ntau = int(rng.integers(0, 3))
        for _t in range(ntau):
            pt = float(rng.exponential(35) + 20); eta = float(rng.uniform(-2.3, 2.3)); phi = float(rng.uniform(-math.pi, math.pi))
            _push(ev, "GenVisTau", pt, eta, phi, 1.0)
            _push(ev, "Tau", pt + rng.normal(0, 1), eta, phi, 1.0)
            ev["Tau_idDeepTau2018v2p5VSjet"].append(6 if rng.random() < tau_eff(pt, eta) else 3)
            ev["Tau_genPartFlav"].append(5)

        gens = []
        for pid, coll in ((11, "Electron"), (13, "Muon")):
            for _l in range(int(rng.integers(0, 3))):
                pt = float(rng.exponential(25) + 10); eta = float(rng.uniform(-2.4, 2.4)); phi = float(rng.uniform(-math.pi, math.pi)); ch = int(rng.choice([-1, 1]))
                mom = len(gens); gens.append((15 * ch, 2, pt + 5, 5.0 * (1 if eta >= 0 else -1), phi, 1.777, -1))
                gens.append((-pid * ch, 1, pt, eta, phi, 0.0, mom))
                if rng.random() < lep_eff(pt, eta):
                    _push(ev, coll, pt, eta, phi); ev[f"{coll}_charge"].append(ch)
        for pdg, st, pt, eta, phi, m, mo in gens:
            ev["GenPart_pdgId"].append(pdg); ev["GenPart_status"].append(st)
            ev["GenPart_pt"].append(pt); ev["GenPart_eta"].append(eta); ev["GenPart_phi"].append(phi)
            ev["GenPart_mass"].append(m); ev["GenPart_genPartIdxMother"].append(mo)

        gmx, gmy = rng.normal(0, 30), rng.normal(0, 30)
        mx, my = gmx + rng.normal(0, met_resolution_gev), gmy + rng.normal(0, met_resolution_gev)
        scal["PuppiMET_pt"].append(math.hypot(mx, my)); scal["PuppiMET_phi"].append(math.atan2(my, mx))
        scal["GenMET_pt"].append(math.hypot(gmx, gmy)); scal["GenMET_phi"].append(math.atan2(gmy, gmx))
        scal["genWeight"].append(1.0)
        for k in cols:
            cols[k].append(ev[k])

    payload = {k: ak.Array(v) for k, v in cols.items()}
    payload.update({k: np.asarray(v, dtype=np.float32) for k, v in scal.items()})
    with uproot.recreate(path) as f:
        f["Events"] = payload
    return NanoTruth(n_events, btag_eff, ctag_eff, ltag_mistag, tau_eff, lep_eff, tau_mistag=tau_mistag)


def _push(ev, coll, pt, eta, phi, mass=None):
    ev[f"{coll}_pt"].append(float(pt)); ev[f"{coll}_eta"].append(float(eta)); ev[f"{coll}_phi"].append(float(phi))
    if mass is not None and f"{coll}_mass" in ev:
        ev[f"{coll}_mass"].append(float(mass))
