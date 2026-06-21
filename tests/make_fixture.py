"""Synthetic Delphes-like ROOT fixture with *injected* object efficiencies.

This is the test oracle. It writes a tree with the same dotted-branch structure
as real Delphes output (collections written as zipped records so ``tree["Jet.PT"]``
resolves) and injects known efficiencies, so the Level-0 measurement can be
checked against ground truth and the Pilot Gate against a deliberately broken
file.

``make_fixture`` returns a :class:`FixtureTruth` carrying the injected efficiency
functions so tests can compute the expected per-bin values without re-deriving
them. Object kinematics are generous (high multiplicity) so every (pT, |eta|)
bin is well populated at a few thousand events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import awkward as ak
import numpy as np
import uproot

Eff = Callable[[float, float], float]

# Default injected efficiencies (smooth, well above zero so bins are populated).
DEF_BTAG_EFF: Eff = lambda pt, eta: 0.55 + 0.15 * np.tanh(pt / 120.0)
DEF_CTAG_EFF: Eff = lambda pt, eta: 0.18 + 0.05 * np.tanh(pt / 150.0)
DEF_LTAG_MISTAG: Eff = lambda pt, eta: 0.010 + 0.00005 * pt
DEF_TAU_EFF: Eff = lambda pt, eta: 0.60 * (abs(eta) < 2.3)
DEF_TAU_MISTAG: Eff = lambda pt, eta: 0.01
DEF_ELE_EFF: Eff = lambda pt, eta: 0.95 * (abs(eta) < 2.5) * (pt > 7.0)
DEF_MUO_EFF: Eff = lambda pt, eta: 0.95 * (abs(eta) < 2.4) * (pt > 5.0)


@dataclass
class FixtureTruth:
    n_events: int
    met_resolution_gev: float
    mbb_width_gev: float
    neg_weight_fraction: float
    btag_eff: Eff
    ctag_eff: Eff
    ltag_mistag: Eff
    tau_eff: Eff
    tau_mistag: Eff
    electron_eff: Eff
    muon_eff: Eff


def make_fixture(
    path: str,
    *,
    n_events: int = 4000,
    seed: int = 0,
    broken: bool = False,
    met_resolution_gev: float = 20.0,
    met_bias_gev: float = 0.0,
    mbb_width_gev: float = 10.0,
    btag_eff: Eff = DEF_BTAG_EFF,
    ctag_eff: Eff = DEF_CTAG_EFF,
    ltag_mistag: Eff = DEF_LTAG_MISTAG,
    tau_eff: Eff = DEF_TAU_EFF,
    tau_mistag: Eff = DEF_TAU_MISTAG,
    electron_eff: Eff = DEF_ELE_EFF,
    muon_eff: Eff = DEF_MUO_EFF,
) -> FixtureTruth:
    """Write a Delphes-like ROOT file to ``path``; return the injected truth.

    ``broken=True`` writes a file that must FAIL the Pilot Gate: a >50% negative
    weight fraction, all-zero ``Jet.Flavor`` (flavour association did not run),
    and no gen taus.
    """
    rng = np.random.default_rng(seed)
    neg_frac = 0.55 if broken else 0.02

    jet_l, ele_l, muo_l, gen_l = [], [], [], []
    met_l, genmet_l, sht_l, ev_l = [], [], [], []
    genjet_l = []

    for _ in range(n_events):
        jets, gens = [], []

        # --- signal Higgs -> bb resonance (gives the m_bb peak the gate checks) ---
        for bpt, beta, bphi, bmass in _resonance_bb(rng, mbb_width_gev):
            flavor = 0 if broken else 5
            p = float(btag_eff(bpt, beta)) if flavor == 5 else float(ltag_mistag(bpt, beta))
            jets.append(_jet(bpt, beta, bphi, flavor, int(rng.random() < p),
                             int(rng.random() < float(tau_mistag(bpt, beta))), mass=bmass))

        # --- extra b/c/light jets (Jet.Flavor drives the b-tag measurement) ------
        njet = int(rng.integers(1, 5))
        for _j in range(njet):
            pt = float(rng.exponential(28.0) + 18.0)
            eta = float(rng.uniform(-2.5, 2.5))
            phi = float(rng.uniform(-math.pi, math.pi))
            flavor = int(rng.choice([5, 4, 0], p=[0.30, 0.20, 0.50]))
            if broken:
                flavor = 0  # flavour association "did not run"
            if flavor == 5:
                p = float(btag_eff(pt, eta))
            elif flavor == 4:
                p = float(ctag_eff(pt, eta))
            else:
                p = float(ltag_mistag(pt, eta))
            btag = int(rng.random() < p)
            tautag = int(rng.random() < float(tau_mistag(pt, eta)))  # jet->tau fake
            jets.append(_jet(pt, eta, phi, flavor, btag, tautag))

        # --- true hadronic taus + their reco tau-jets ----------------------
        # A hadronic tau always yields a reco jet near it; whether that jet is
        # tau-tagged is the efficiency, and it carries a real light b-mistag draw.
        if not broken:
            ntau = int(rng.integers(0, 3))
            for _t in range(ntau):
                pt = float(rng.exponential(35.0) + 20.0)
                eta = float(rng.uniform(-2.4, 2.4))
                phi = float(rng.uniform(-math.pi, math.pi))
                gens.append(_gen(pid=15 * rng.choice([-1, 1]), status=2,
                                 pt=pt, eta=eta, phi=phi, mass=1.777))
                jpt = pt + float(rng.normal(0, 1.0))
                jets.append(_jet(jpt, eta, phi, flavor=0,
                                 btag=int(rng.random() < float(ltag_mistag(jpt, eta))),
                                 tautag=int(rng.random() < float(tau_eff(pt, eta)))))

        # --- electrons / muons (gen + reco at injected efficiency) ---------
        # In the good (non-broken) fixture each prompt lepton is given a tau
        # mother so the gen.m1 chain points to |PID|==15, mirroring HH→ττ→ℓνν
        # and matching the prompt-mother selection in
        # validation/level0_objects/leptons.py. The mother tau is parked at
        # |eta|=5.0 (well outside Delphes' TauEtaMax = 2.5 and any jet's
        # acceptance) so it cannot be unique-matched to a jet by tau.py and
        # therefore cannot pollute the tau efficiency or jet→τ mistag
        # measurements. The broken fixture skips mother insertion to keep the
        # gen-tau count at zero (the Pilot Gate must fail there).
        for pid, eff, sink in ((11, electron_eff, ele_l), (13, muon_eff, muo_l)):
            nlep = int(rng.integers(0, 3))
            recos = []
            for _l in range(nlep):
                pt = float(rng.exponential(25.0) + 7.0)
                eta = float(rng.uniform(-2.6, 2.6))
                phi = float(rng.uniform(-math.pi, math.pi))
                charge = int(rng.choice([-1, 1]))
                if broken:
                    mother_idx = -1
                else:
                    mother_idx = len(gens)
                    gens.append(_gen(pid=15 * charge, status=2,
                                     pt=pt + 5.0,
                                     eta=5.0 * (1 if eta >= 0 else -1),
                                     phi=phi, mass=1.777))
                gens.append(_gen(pid=-pid * charge, status=1,
                                 pt=pt, eta=eta, phi=phi, mass=0.0,
                                 m1=mother_idx))
                if rng.random() < float(eff(pt, eta)):
                    recos.append(_lep(pt, eta, phi, charge))
            sink.append(recos)

        # --- MET: reco = gen smeared by a known resolution -----------------
        gmx, gmy = rng.normal(0, 30.0), rng.normal(0, 30.0)
        mx = gmx + met_bias_gev + rng.normal(0, met_resolution_gev)
        my = gmy + met_bias_gev + rng.normal(0, met_resolution_gev)
        met_l.append([{"MET": math.hypot(mx, my), "Eta": 0.0, "Phi": math.atan2(my, mx)}])
        genmet_l.append([{"MET": math.hypot(gmx, gmy), "Eta": 0.0, "Phi": math.atan2(gmy, gmx)}])

        sht_l.append([{"HT": float(sum(j["PT"] for j in jets))}])
        w = -1.0 if rng.random() < neg_frac else 1.0
        ev_l.append([{"Weight": w}])
        genjet_l.append([{"PT": j["PT"], "Eta": j["Eta"], "Phi": j["Phi"],
                          "Mass": j["Mass"], "Flavor": j["Flavor"]} for j in jets[:2]])

        jet_l.append(jets)
        gen_l.append(gens)

    payload = {
        "Jet": ak.Array(jet_l),
        "GenJet": ak.Array(genjet_l),
        "Electron": ak.Array(ele_l),
        "Muon": ak.Array(muo_l),
        "Particle": ak.Array(gen_l),
        "MissingET": ak.Array(met_l),
        "GenMissingET": ak.Array(genmet_l),
        "ScalarHT": ak.Array(sht_l),
        "Event": ak.Array(ev_l),
    }
    with uproot.recreate(path) as f:
        f["Delphes"] = payload

    return FixtureTruth(
        n_events=n_events,
        met_resolution_gev=met_resolution_gev,
        mbb_width_gev=mbb_width_gev,
        neg_weight_fraction=neg_frac,
        btag_eff=btag_eff,
        ctag_eff=ctag_eff,
        ltag_mistag=ltag_mistag,
        tau_eff=tau_eff,
        tau_mistag=tau_mistag,
        electron_eff=electron_eff,
        muon_eff=muon_eff,
    )


# --- record builders (consistent keys per collection) --------------------- #
def _jet(pt, eta, phi, flavor, btag, tautag, mass=8.0):
    return {"PT": float(pt), "Eta": float(eta), "Phi": float(phi), "Mass": float(mass),
            "Flavor": int(flavor), "BTag": int(btag), "TauTag": int(tautag),
            "Charge": 0}


def _boost(vec, beta):
    """Active Lorentz boost of 4-vector ``vec=(E,px,py,pz)`` by velocity ``beta``."""
    e, px, py, pz = vec
    bx, by, bz = beta
    b2 = bx * bx + by * by + bz * bz
    if b2 <= 0.0:
        return vec
    gamma = 1.0 / math.sqrt(1.0 - b2)
    bp = bx * px + by * py + bz * pz
    g2 = (gamma - 1.0) / b2
    return (
        gamma * (e + bp),
        px + g2 * bp * bx + gamma * bx * e,
        py + g2 * bp * by + gamma * by * e,
        pz + g2 * bp * bz + gamma * bz * e,
    )


def _resonance_bb(rng, width):
    """Two b-jets from a Higgs -> bb decay with exact pair mass ~ N(125, width).

    Returns ``[(pt, eta, phi, mass), (pt, eta, phi, mass)]``; the invariant mass
    of the pair equals the sampled m_H by construction (back-to-back decay in the
    rest frame, then boosted to the lab)."""
    mh = max(float(rng.normal(125.0, width)), 30.0)
    mb = 4.18
    e = mh / 2.0
    pmag = math.sqrt(max(e * e - mb * mb, 0.0))
    ct = rng.uniform(-1.0, 1.0)
    st = math.sqrt(max(1.0 - ct * ct, 0.0))
    ph = rng.uniform(-math.pi, math.pi)
    d = (st * math.cos(ph), st * math.sin(ph), ct)
    rest = [
        (e, pmag * d[0], pmag * d[1], pmag * d[2]),
        (e, -pmag * d[0], -pmag * d[1], -pmag * d[2]),
    ]
    # Higgs lab momentum
    pth = float(rng.exponential(70.0))
    etah = float(rng.uniform(-2.0, 2.0))
    phih = float(rng.uniform(-math.pi, math.pi))
    pxh, pyh, pzh = pth * math.cos(phih), pth * math.sin(phih), pth * math.sinh(etah)
    eh = math.sqrt(pxh * pxh + pyh * pyh + pzh * pzh + mh * mh)
    beta = (pxh / eh, pyh / eh, pzh / eh)
    out = []
    for vr in rest:
        _e, px, py, pz = _boost(vr, beta)
        pt = math.hypot(px, py)
        eta = math.asinh(pz / pt) if pt > 1e-6 else 0.0
        out.append((pt, eta, math.atan2(py, px), mb))
    return out


def _lep(pt, eta, phi, charge):
    return {"PT": float(pt), "Eta": float(eta), "Phi": float(phi), "Charge": int(charge)}


def _gen(pid, status, pt, eta, phi, mass, m1=-1):
    return {"PID": int(pid), "Status": int(status), "PT": float(pt), "Eta": float(eta),
            "Phi": float(phi), "Mass": float(mass), "Charge": 0,
            "M1": int(m1), "M2": -1, "D1": -1, "D2": -1}


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/signal_fixture.root"
    truth = make_fixture(out)
    print(f"wrote {out}: {truth.n_events} events, "
          f"neg-weight frac {truth.neg_weight_fraction}")
