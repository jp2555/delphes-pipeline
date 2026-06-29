"""Synthetic HH→bb̄ττ signal fixture for the Level-4 (κ_λ m_HH) tests.

Generates a full decay chain — HH → H(bb̄) + H(τ⁺τ⁻), each H at 125 GeV — so the gen
record carries the two Higgs / two b-quarks / two τ's and the reco objects (b-jets,
τ_h jets / leptons, pᵀᵐⁱˢˢ) track them with a *controllable* m_HH scale (``bjet_response``)
and resolution (``bjet_smear``). gen m_HH and reco m_HH can then be compared.
"""

from __future__ import annotations

import math

import awkward as ak
import numpy as np
import uproot
from make_candle_fixture import _jet, _lep, _visfrac
from make_fixture import _boost


def _mass(p4):
    e, px, py, pz = p4
    return math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))


def _ptetaphi(p4):
    _e, px, py, pz = p4
    pt = math.hypot(px, py)
    return pt, (math.asinh(pz / pt) if pt > 1e-6 else 0.0), math.atan2(py, px)


def _two_body(rng, parent, m1, m2):
    """Decay ``parent`` (E,px,py,pz) into two daughters of mass m1,m2, isotropic in the
    parent rest frame, boosted back to the lab."""
    M = _mass(parent)
    e1 = (M * M + m1 * m1 - m2 * m2) / (2 * M)
    p = math.sqrt(max(e1 * e1 - m1 * m1, 0.0))
    ct = rng.uniform(-1, 1); st = math.sqrt(max(1 - ct * ct, 0.0)); ph = rng.uniform(-math.pi, math.pi)
    d = (st * math.cos(ph), st * math.sin(ph), ct)
    rest1 = (e1, p * d[0], p * d[1], p * d[2])
    rest2 = (M - e1, -p * d[0], -p * d[1], -p * d[2])
    beta = (parent[1] / parent[0], parent[2] / parent[0], parent[3] / parent[0])
    return _boost(rest1, beta), _boost(rest2, beta)


def _gen(pid, p4, *, m1=-1, status=22):
    pt, eta, phi = _ptetaphi(p4)
    return {"PID": int(pid), "Status": int(status), "PT": float(pt), "Eta": float(eta), "Phi": float(phi),
            "Mass": float(_mass(p4)), "Charge": 0, "M1": int(m1), "M2": -1, "D1": -1, "D2": -1}


def make_hhbbtt_fixture(path, *, n_events=3000, seed=0, mhh_lo=250.0, mhh_hi=800.0,
                        bjet_response=1.0, bjet_smear=0.0, met_res=12.0, add_copies=False) -> None:
    rng = np.random.default_rng(seed)
    jets, ele, muo, gen, met, genmet = [], [], [], [], [], []
    for _ in range(n_events):
        js, gs, es, ms = [], [], [], []
        mhh = rng.uniform(mhh_lo, mhh_hi)
        # the HH system with some recoil pT
        hh_pt = float(rng.exponential(40)); hh_eta = rng.uniform(-2, 2); hh_phi = rng.uniform(-math.pi, math.pi)
        px, py, pz = hh_pt * math.cos(hh_phi), hh_pt * math.sin(hh_phi), hh_pt * math.sinh(hh_eta)
        hh = (math.sqrt(px * px + py * py + pz * pz + mhh * mhh), px, py, pz)
        h_bb, h_tt = _two_body(rng, hh, 125.0, 125.0)
        i_hbb = len(gs); gs.append(_gen(25, h_bb))
        i_htt = len(gs); gs.append(_gen(25, h_tt))

        # H -> b b̄: gen quarks (mother = the bb Higgs) + reco b-jets (reco pT = gen × response × smear)
        b_pair = _two_body(rng, h_bb, 4.7, 4.7)
        for b in b_pair:
            gs.append(_gen(5, b, m1=i_hbb))
            pt, eta, phi = _ptetaphi(b)
            reco_pt = pt * bjet_response * (1.0 + rng.normal(0, bjet_smear))
            js.append(_jet(reco_pt, eta, phi, flavor=5, btag=1, mass=8.0))

        # H -> τ τ: gen τ's (mother = the ττ Higgs) + visible legs (x) with neutrinos -> reco τ_h/lepton + MET
        leptonic = rng.random() < 0.5
        nu_x = nu_y = 0.0
        tau_pair = _two_body(rng, h_tt, 1.777, 1.777)
        for i, t in enumerate(tau_pair):
            gs.append(_gen(15 * (1 if i == 0 else -1), t, m1=i_htt))
            _e, tx, ty, tz = t
            is_lep = leptonic and i == 0
            x = _visfrac(rng, is_lep)
            vx, vy, vz = tx * x, ty * x, tz * x
            pt_v = math.hypot(vx, vy); eta_v = math.asinh(vz / pt_v) if pt_v > 1e-6 else 0.0
            phi_v = math.atan2(vy, vx)
            nu_x += tx * (1 - x); nu_y += ty * (1 - x)
            if is_lep:
                (es if rng.random() < 0.5 else ms).append(_lep(pt_v, eta_v, phi_v, -1 if i == 0 else 1))
            else:
                js.append(_jet(pt_v, eta_v, phi_v, flavor=0, btag=0, tautag=1, mass=float(rng.uniform(0.6, 1.5))))
        if add_copies:
            # mimic the full Pythia record: extra |pid| copies of the b's/τ's (which would
            # fool a leading-pT b/τ selection) + low-status Higgs copies (must be filtered
            # by status>=20). gen m_HH (Higgs-based) must ignore all of these.
            for b in b_pair:
                gs.append(_gen(5, b))                                  # duplicate hard b
            gs.append(_gen(5, (b_pair[0][0] * 0.4, b_pair[0][1] * 0.4,  # a soft radiated b
                               b_pair[0][2] * 0.4, b_pair[0][3] * 0.4)))
            for t in tau_pair:
                gs.append(_gen(15, t))                                 # duplicate hard τ
            for h in (h_bb, h_tt):
                c = _gen(25, h); c["Status"] = 5; gs.append(c)         # low-status Higgs copy

        mx, my = nu_x + rng.normal(0, met_res), nu_y + rng.normal(0, met_res)
        met.append([{"MET": math.hypot(mx, my), "Eta": 0.0, "Phi": math.atan2(my, mx)}])
        genmet.append([{"MET": math.hypot(nu_x, nu_y), "Eta": 0.0, "Phi": math.atan2(nu_y, nu_x)}])
        jets.append(js); gen.append(gs); ele.append(es); muo.append(ms)
    if not any(ele):
        ele[0] = [_lep(30, 0, 0, 1)]
    if not any(muo):
        muo[0] = [_lep(30, 0, 0, 1)]

    payload = {
        "Jet": ak.Array(jets), "Electron": ak.Array(ele), "Muon": ak.Array(muo),
        "Particle": ak.Array(gen), "MissingET": ak.Array(met), "GenMissingET": ak.Array(genmet),
        "Event": ak.Array([[{"Weight": 1.0}] for _ in range(n_events)]),
        "ScalarHT": ak.Array([[{"HT": float(sum(j["PT"] for j in ev))}] for ev in jets]),
    }
    with uproot.recreate(path) as f:
        f["Delphes"] = payload
