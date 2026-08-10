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
