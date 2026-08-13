"""MET smearing: put back the resolution the no-pileup card cannot produce.

The card runs without pileup by design (D3 option A) and its header delegates pileup to
the tuning maps. Delphes MET is therefore unphysically clean (~16 GeV per component vs
~33 on the CMS anchor), and MET is exactly what FastMTT fits the τ energy fractions
against — so the gap propagates into m_ττ and everything built from the di-τ system.
"""

from __future__ import annotations

from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

from delphes_pipeline.tuning.maps import RetaggedEvents, TuningMaps, smear_met

_N = 20000
_TRUE = 40.0          # per-component "true" MET the fixture starts from


def _events(seed=3, res=16.0):
    """Events whose MET already carries a per-component resolution of ``res``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, res, _N) + _TRUE
    y = rng.normal(0.0, res, _N)
    met = ak.zip({"met": np.hypot(x, y), "eta": np.zeros(_N), "phi": np.arctan2(y, x)})
    gen = ak.zip({"met": np.full(_N, _TRUE), "eta": np.zeros(_N), "phi": np.zeros(_N)})
    return SimpleNamespace(met=met, genmet=gen)


def _resolution(ev):
    """Per-component RMS of (reco - gen), the estimator the maps are derived with."""
    dx = ak.to_numpy(ev.met.met * np.cos(ev.met.phi)) - _TRUE
    dy = ak.to_numpy(ev.met.met * np.sin(ev.met.phi))
    return float(np.sqrt(0.5 * (np.var(dx) + np.var(dy))))


def _maps(sigma):
    return TuningMaps({"met_smear": {"x": "overall", "centers": [0.0],
                                     "values": [sigma], "counts": [_N]}})


def test_smearing_reaches_the_anchor_resolution():
    """σ is the quadrature gap, so the smeared resolution lands on the target."""
    ev = _events(res=16.0)
    target = 32.5
    sigma = np.sqrt(target ** 2 - _resolution(ev) ** 2)
    ev.met = smear_met(ev, _maps(sigma), np.random.default_rng(0))
    assert _resolution(ev) == pytest.approx(target, rel=0.03)


def test_zero_sigma_is_a_no_op():
    ev = _events()
    before = ak.to_numpy(ev.met.met).copy()
    out = smear_met(ev, _maps(0.0), np.random.default_rng(0))
    assert np.allclose(ak.to_numpy(out.met), before)


def test_absent_map_leaves_met_untouched():
    ev = _events()
    before = ak.to_numpy(ev.met.met).copy()
    out = smear_met(ev, TuningMaps({}), np.random.default_rng(0))
    assert np.allclose(ak.to_numpy(out.met), before)


def test_same_seed_gives_identical_met():
    """The lens and the ntuplizer must see the same smeared MET."""
    a = smear_met(_events(), _maps(28.0), np.random.default_rng(0))
    b = smear_met(_events(), _maps(28.0), np.random.default_rng(0))
    assert np.allclose(ak.to_numpy(a.met), ak.to_numpy(b.met))


def test_retagged_events_exposes_the_smeared_met():
    ev = _events()
    ev.jets = ak.zip({k: ak.Array([[]] * _N) for k in
                      ("pt", "eta", "phi", "mass", "btag", "tautag", "flavor")})
    view = RetaggedEvents(ev, _maps(28.0), np.random.default_rng(0))
    assert "met_smear" in view.retagged_fields
    assert not np.allclose(ak.to_numpy(view.met.met), ak.to_numpy(ev.met.met))
    assert _resolution(view) > _resolution(ev)


def test_retagged_events_without_the_map_passes_met_through():
    ev = _events()
    ev.jets = ak.zip({k: ak.Array([[]] * _N) for k in
                      ("pt", "eta", "phi", "mass", "btag", "tautag", "flavor")})
    view = RetaggedEvents(ev, TuningMaps({}), np.random.default_rng(0))
    assert "met_smear" not in view.retagged_fields
    assert np.allclose(ak.to_numpy(view.met.met), ak.to_numpy(ev.met.met))


# --------------------------------------------------------------------------- #
# HT-dependent smearing. One flat width matches the anchor only ON AVERAGE:
# measured, CMS rises +2.83 GeV per 100 GeV of jet HT against a flat +0.27, so a
# single number over-smears quiet events by ~57% and under-smears busy ones. And
# because m_ττ = m_vis/√(x₁x₂) is nonlinear, excess pT_miss noise moves the m_ττ
# MEAN, not only its width.
# --------------------------------------------------------------------------- #
def _ht_events(n=6000, seed=0):
    """Half quiet (HT~150), half busy (HT~600), with clean gen MET."""
    rng = np.random.default_rng(seed)
    jpt = np.where(np.arange(n) % 2 == 0, 75.0, 300.0)
    jets = ak.zip({"pt": ak.Array([[p, p] for p in jpt]),
                   "eta": ak.Array([[0.3, -0.4]] * n),
                   "phi": ak.Array([[0.2, 2.4]] * n),
                   "mass": ak.Array([[5.0, 5.0]] * n),
                   "btag": ak.Array([[0, 0]] * n), "tautag": ak.Array([[0, 0]] * n)})
    g = rng.normal(0, 1e-6, n)
    met = ak.zip({"met": np.abs(g) + 30.0, "eta": np.zeros(n),
                  "phi": np.full(n, 0.7)})
    return SimpleNamespace(jets=jets, met=met, genmet=met, n=n), jpt * 2


def _ht_map(lo_sigma, hi_sigma):
    return TuningMaps({"met_smear": {"x": "ht", "centers": [150.0, 600.0],
                                     "values": [lo_sigma, hi_sigma],
                                     "counts": [3000, 3000]}})


def _width(a):
    lo, hi = np.percentile(a, [16, 84])
    return 0.5 * (hi - lo)


def test_sigma_follows_ht_when_the_map_is_a_curve():
    ev, ht = _ht_events()
    out = smear_met(ev, _ht_map(5.0, 40.0), np.random.default_rng(0))
    x = ak.to_numpy(out.met * np.cos(out.phi)) - 30.0 * np.cos(0.7)
    quiet, busy = _width(x[ht < 300]), _width(x[ht > 300])
    assert quiet == pytest.approx(5.0, rel=0.25), quiet
    assert busy == pytest.approx(40.0, rel=0.15), busy
    assert busy / quiet > 4.0, "a flat map would give ratio 1"


def test_a_flat_map_smears_quiet_and_busy_events_identically():
    """The defect this replaces: one width for every event."""
    ev, ht = _ht_events(seed=2)
    flat = TuningMaps({"met_smear": {"x": "overall", "centers": [0.0], "values": [22.0]}})
    out = smear_met(ev, flat, np.random.default_rng(0))
    x = ak.to_numpy(out.met * np.cos(out.phi)) - 30.0 * np.cos(0.7)
    assert _width(x[ht < 300]) / _width(x[ht > 300]) == pytest.approx(1.0, abs=0.12)


def test_single_value_maps_keep_the_old_flat_behaviour():
    """Maps derived before this change must still work, unchanged."""
    ev, _ = _ht_events(seed=3)
    legacy = TuningMaps({"met_smear": {"x": "overall", "centers": [0.0], "values": [17.0]}})
    out = smear_met(ev, legacy, np.random.default_rng(5))
    x = ak.to_numpy(out.met * np.cos(out.phi)) - 30.0 * np.cos(0.7)
    assert _width(x) == pytest.approx(17.0, rel=0.15)


def test_ht_is_computed_from_jets_not_from_scalar_ht():
    """ScalarHT and NanoAOD sumEt are different variables; binning by them would compare
    different things across tiers, which is why the anchor map used to be a single bin."""
    from delphes_pipeline.core import observables as obs

    ev, ht = _ht_events(seed=4)
    ev.scalar_ht = ak.zip({"ht": ak.Array(np.full(ev.n, 9999.0))})   # deliberately wrong
    assert obs.jet_ht(ev) == pytest.approx(ht)


def test_derived_map_makes_a_flat_delphes_track_a_rising_anchor():
    """The closure the whole change exists for: solve the quadrature gap per HT bin and
    the smeared Delphes resolution must follow the anchor's RISE, not just its average."""
    from delphes_pipeline.core import observables as obs

    rng = np.random.default_rng(0)
    n = 40000
    jpt = rng.choice([60.0, 110.0, 200.0, 380.0], n)
    jets = ak.zip({"pt": ak.Array([[p, p] for p in jpt]),
                   "eta": ak.Array([[0.3, -0.4]] * n), "phi": ak.Array([[0.2, 2.4]] * n),
                   "mass": ak.Array([[5.0, 5.0]] * n)})
    mk = lambda x, y: ak.zip({"met": np.hypot(x, y), "eta": np.zeros(n),
                              "phi": np.arctan2(y, x)})
    g = rng.normal(0, 1e-9, (2, n))
    d = rng.normal(0, 16.0, (2, n))                      # Delphes: flat 16 GeV
    ev = SimpleNamespace(jets=jets, met=mk(g[0] + d[0], g[1] + d[1]),
                         genmet=mk(*g), n=n)
    prof = obs.met_resolution_vs_ht(ev)
    want = 12.0 + 0.05 * np.asarray(prof.centers)        # a CMS-like rise
    sig = np.sqrt(np.maximum(want ** 2 - np.asarray(prof.values) ** 2, 0.0))
    m = TuningMaps({"met_smear": {"x": "ht", "centers": np.asarray(prof.centers).tolist(),
                                  "values": sig.tolist(),
                                  "counts": np.asarray(prof.counts).tolist()}})
    ev.met = smear_met(ev, m, np.random.default_rng(1))
    got = obs.met_resolution_vs_ht(ev)
    assert np.allclose(np.asarray(got.values), want, rtol=0.05), (got.values, want)


def test_resolution_vs_ht_does_not_need_scalar_ht():
    """It must not require the very branch whose cross-tier incomparability is the reason
    this measurement is binned in jet HT instead."""
    from delphes_pipeline.core import observables as obs

    ev, _ = _ht_events(seed=9)
    assert not hasattr(ev, "scalar_ht")
    prof = obs.met_resolution_vs_ht(ev)
    assert np.asarray(prof.values).size >= 1
