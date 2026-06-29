"""Synthetic tt̄ and Z→ττ Delphes fixtures for the Level-1 candle tests.

``make_ttbar_fixture`` writes eμ events with exactly two b-jets tagged at an
injected efficiency (so the in-situ ε_b recovers it). ``make_dy_fixture`` writes a
visible di-τ resonance (τ_hτ_h / ℓτ_h, no b-jets) at a target visible mass.
"""

from __future__ import annotations

import math

import awkward as ak
import numpy as np
import uproot
from make_fixture import _boost


def _jet(pt, eta, phi, flavor, btag, tautag=0, mass=8.0):
    return {"PT": float(pt), "Eta": float(eta), "Phi": float(phi), "Mass": float(mass),
            "Flavor": int(flavor), "BTag": int(btag), "TauTag": int(tautag), "Charge": 0}


def _lep(pt, eta, phi, charge):
    return {"PT": float(pt), "Eta": float(eta), "Phi": float(phi), "Charge": int(charge)}


def _met_records(rng, res=22.0):
    gmx, gmy = rng.normal(0, 30), rng.normal(0, 30)
    mx, my = gmx + rng.normal(0, res), gmy + rng.normal(0, res)
    return ([{"MET": math.hypot(mx, my), "Eta": 0.0, "Phi": math.atan2(my, mx)}],
            [{"MET": math.hypot(gmx, gmy), "Eta": 0.0, "Phi": math.atan2(gmy, gmx)}])


def _write(path, *, jets, electrons, muons, gen, met, genmet):
    n = len(jets)
    payload = {}
    # an all-empty (fieldless) collection cannot be typed by uproot -> omit it;
    # the reader treats a missing branch as a per-event empty collection.
    for name, coll in (("Jet", jets), ("Electron", electrons), ("Muon", muons), ("Particle", gen)):
        if any(len(ev) for ev in coll):
            payload[name] = ak.Array(coll)
    payload["MissingET"] = ak.Array(met)
    payload["GenMissingET"] = ak.Array(genmet)
    payload["Event"] = ak.Array([[{"Weight": 1.0}] for _ in range(n)])
    payload["ScalarHT"] = ak.Array([[{"HT": float(sum(j["PT"] for j in js))}] for js in jets])
    with uproot.recreate(path) as f:
        f["Delphes"] = payload


def make_ttbar_fixture(path, *, n_events=3000, seed=0, btag_eff=0.70) -> float:
    rng = np.random.default_rng(seed)
    jets, ele, muo, gen, met, genmet = [], [], [], [], [], []
    for _ in range(n_events):
        js = []
        for _b in range(2):  # exactly two b-jets
            pt = float(rng.exponential(50) + 30)
            js.append(_jet(pt, rng.uniform(-2.3, 2.3), rng.uniform(-math.pi, math.pi),
                           flavor=5, btag=int(rng.random() < btag_eff)))
        for _l in range(int(rng.integers(0, 3))):  # a few light jets, b-veto-irrelevant
            pt = float(rng.exponential(25) + 20)
            js.append(_jet(pt, rng.uniform(-2.5, 2.5), rng.uniform(-math.pi, math.pi),
                           flavor=0, btag=int(rng.random() < 0.01)))
        ch = int(rng.choice([-1, 1]))
        ele.append([_lep(rng.exponential(25) + 20, rng.uniform(-2.4, 2.4), rng.uniform(-math.pi, math.pi), ch)])
        muo.append([_lep(rng.exponential(25) + 20, rng.uniform(-2.4, 2.4), rng.uniform(-math.pi, math.pi), -ch)])
        gen.append([])
        m, gm = _met_records(rng, res=28.0)
        met.append(m); genmet.append(gm); jets.append(js)
    _write(path, jets=jets, electrons=ele, muons=muo, gen=gen, met=met, genmet=genmet)
    return btag_eff


def make_dy_fixture(path, *, n_events=3000, seed=0, vis_mass=65.0, width=8.0) -> float:
    rng = np.random.default_rng(seed)
    jets, ele, muo, gen, met, genmet = [], [], [], [], [], []
    for _ in range(n_events):
        js, gs, es, ms = [], [], [], []
        leptonic = rng.random() < 0.5  # half ℓτ_h, half τ_hτ_h
        for i, (pt, eta, phi) in enumerate(_ditau(rng, vis_mass, width)):
            gs.append({"PID": 15 * (1 if i == 0 else -1), "Status": 2, "PT": float(pt), "Eta": float(eta),
                       "Phi": float(phi), "Mass": 1.777, "Charge": 0, "M1": -1, "M2": -1, "D1": -1, "D2": -1})
            if leptonic and i == 0:  # one leg as a lepton
                (es if rng.random() < 0.5 else ms).append(_lep(pt, eta, phi, int(rng.choice([-1, 1]))))
            else:
                js.append(_jet(pt, eta, phi, flavor=0, btag=0, tautag=1))
        jets.append(js); gen.append(gs); ele.append(es); muo.append(ms)
        m, gm = _met_records(rng, res=20.0)
        met.append(m); genmet.append(gm)
    # ensure Electron/Muon branches are typed even if a run produced none
    if not any(ele):
        ele[0] = [_lep(30, 0, 0, 1)]
    if not any(muo):
        muo[0] = [_lep(30, 0, 0, 1)]
    _write(path, jets=jets, electrons=ele, muons=muo, gen=gen, met=met, genmet=genmet)
    return vis_mass


def _ditau(rng, mass, width):
    mh = max(float(rng.normal(mass, width)), 20.0)
    mtau = 1.777
    e = mh / 2.0
    p = math.sqrt(max(e * e - mtau * mtau, 0.0))
    ct = rng.uniform(-1, 1)
    st = math.sqrt(max(1 - ct * ct, 0.0))
    ph = rng.uniform(-math.pi, math.pi)
    d = (st * math.cos(ph), st * math.sin(ph), ct)
    rest = [(e, p * d[0], p * d[1], p * d[2]), (e, -p * d[0], -p * d[1], -p * d[2])]
    pth = float(rng.exponential(35))
    etah, phih = rng.uniform(-1.5, 1.5), rng.uniform(-math.pi, math.pi)
    pxh, pyh, pzh = pth * math.cos(phih), pth * math.sin(phih), pth * math.sinh(etah)
    eh = math.sqrt(pxh**2 + pyh**2 + pzh**2 + mh**2)
    beta = (pxh / eh, pyh / eh, pzh / eh)
    out = []
    for vr in rest:
        _e, px, py, pz = _boost(vr, beta)
        pt = math.hypot(px, py)
        out.append((pt, math.asinh(pz / pt) if pt > 1e-6 else 0.0, math.atan2(py, px)))
    return out


def _ditau_taus(rng, mass, width):
    """Two boosted τ 4-vectors ``(E,px,py,pz)`` with di-τ invariant mass ~``mass``."""
    mh = max(float(rng.normal(mass, width)), 40.0)
    mtau = 1.777
    e = mh / 2.0
    p = math.sqrt(max(e * e - mtau * mtau, 0.0))
    ct = rng.uniform(-1, 1); st = math.sqrt(max(1 - ct * ct, 0.0)); ph = rng.uniform(-math.pi, math.pi)
    d = (st * math.cos(ph), st * math.sin(ph), ct)
    rest = [(e, p * d[0], p * d[1], p * d[2]), (e, -p * d[0], -p * d[1], -p * d[2])]
    pth = float(rng.exponential(35))
    etah, phih = rng.uniform(-1.5, 1.5), rng.uniform(-math.pi, math.pi)
    pxh, pyh, pzh = pth * math.cos(phih), pth * math.sin(phih), pth * math.sinh(etah)
    eh = math.sqrt(pxh**2 + pyh**2 + pzh**2 + mh**2)
    beta = (pxh / eh, pyh / eh, pzh / eh)
    return [_boost(vr, beta) for vr in rest]


def _visfrac(rng, is_lep):
    """Visible energy fraction x = E_vis/E_τ (lepton carries less; hadron more)."""
    return float(rng.uniform(0.2, 0.6) if is_lep else rng.uniform(0.45, 0.95))


def make_dy_fixture_nu(path, *, n_events=3000, seed=0, true_mass=91.2, width=3.0, met_res=10.0,
                       vis_smear=0.0, acoll=0.0, met_extra=0.0) -> float:
    """Z→ττ with *modelled* τ decays: each visible leg is ``x_i`` × the τ momentum and
    pᵀᵐⁱˢˢ is the Σ neutrino pT (+ resolution). The visible mass sits below ``true_mass``;
    the FastMTT estimator should recover ``true_mass`` (= m_Z by default).

    ``vis_smear`` (relative pT), ``acoll`` (rad, breaks exact collinearity) and
    ``met_extra`` (uncorrelated MET noise) perturb the visible legs/MET *away* from the
    estimator's exact generative model — set them for a non-circular robustness test."""
    rng = np.random.default_rng(seed)
    jets, ele, muo, gen, met, genmet = [], [], [], [], [], []
    for _ in range(n_events):
        js, gs, es, ms = [], [], [], []
        leptonic = rng.random() < 0.5
        nu_x = nu_y = 0.0
        for i, t in enumerate(_ditau_taus(rng, true_mass, width)):
            _e, tx, ty, tz = t
            pt_t = math.hypot(tx, ty)
            pid = 15 * (1 if i == 0 else -1)  # i==0 is τ⁻
            gs.append({"PID": pid, "Status": 2, "PT": float(pt_t),
                       "Eta": math.asinh(tz / pt_t) if pt_t > 1e-6 else 0.0, "Phi": math.atan2(ty, tx),
                       "Mass": 1.777, "Charge": 0, "M1": -1, "M2": -1, "D1": -1, "D2": -1})
            is_lep = leptonic and i == 0
            x = _visfrac(rng, is_lep)
            vx, vy, vz = tx * x, ty * x, tz * x
            pt_v = math.hypot(vx, vy) * (1.0 + rng.normal(0, vis_smear))
            eta_v = (math.asinh(vz / math.hypot(vx, vy)) if math.hypot(vx, vy) > 1e-6 else 0.0) + rng.normal(0, acoll)
            phi_v = math.atan2(vy, vx) + rng.normal(0, acoll)
            nu_x += tx * (1 - x); nu_y += ty * (1 - x)
            if is_lep:
                charge = -1 if pid > 0 else 1  # τ⁻ → ℓ⁻
                (es if rng.random() < 0.5 else ms).append(_lep(pt_v, eta_v, phi_v, charge))
            else:
                js.append(_jet(pt_v, eta_v, phi_v, flavor=0, btag=0, tautag=1, mass=float(rng.uniform(0.6, 1.5))))
        mx = nu_x + rng.normal(0, math.hypot(met_res, met_extra))
        my = nu_y + rng.normal(0, math.hypot(met_res, met_extra))
        met.append([{"MET": math.hypot(mx, my), "Eta": 0.0, "Phi": math.atan2(my, mx)}])
        genmet.append([{"MET": math.hypot(nu_x, nu_y), "Eta": 0.0, "Phi": math.atan2(nu_y, nu_x)}])
        jets.append(js); gen.append(gs); ele.append(es); muo.append(ms)
    if not any(ele):
        ele[0] = [_lep(30, 0, 0, 1)]
    if not any(muo):
        muo[0] = [_lep(30, 0, 0, 1)]
    _write(path, jets=jets, electrons=ele, muons=muo, gen=gen, met=met, genmet=genmet)
    return true_mass
