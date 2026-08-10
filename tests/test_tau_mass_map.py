"""τ_h visible-mass tuning map: the fix for the FastMTT NaN that ate ~89% of events.

A Delphes τ_h is a jet, so it carries the AK4 jet mass (multi-GeV). FastMTT's hadronic
decay prior is zero unless ``(m_vis/m_τ)² ≤ 1`` with m_τ = 1.777 GeV, so a τ-jet keeping
its jet mass admits no solution at all and m_ττ comes back NaN. The map replaces that
mass with the CMS ``Tau`` visible mass measured on the anchor.
"""

from __future__ import annotations

from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

from delphes_pipeline.extensions.mtautau import M_TAU, fastmtt_mass
from delphes_pipeline.tuning.maps import TuningMaps, retag_jets, tau_visible_mass

_N = 50
_CENTERS = [25.0, 45.0, 85.0, 175.0]
_CMS_MASS = [0.75, 0.85, 0.95, 1.05]        # CMS-like τ_h visible masses (all < m_τ)


def _maps(**extra):
    m = {"tau_mass": {"x": "pt", "centers": _CENTERS, "values": _CMS_MASS,
                      "counts": [100] * len(_CENTERS)}}
    m.update(extra)
    return TuningMaps(m)


def _jag(vals):
    return ak.values_astype(ak.Array([list(vals)] * _N), np.float64)


def _jets(pt, mass, tautag):
    return ak.zip({"pt": _jag(pt), "eta": _jag([0.3] * len(pt)), "phi": _jag([0.0] * len(pt)),
                   "mass": _jag(mass), "tautag": _jag(tautag),
                   "btag": _jag([0] * len(pt)), "flavor": _jag([0] * len(pt))})


def test_tau_jets_take_the_anchor_mass_and_other_jets_do_not():
    jets = _jets(pt=[45.0, 45.0], mass=[8.0, 12.0], tautag=[1, 0])
    out = ak.to_numpy(tau_visible_mass(jets, _maps()))
    assert out[0][0] == pytest.approx(0.85, abs=1e-6)   # τ-jet -> anchor mass at 45 GeV
    assert out[0][1] == pytest.approx(12.0, abs=1e-6)   # non-τ jet untouched


def test_assigned_mass_is_always_below_m_tau():
    """Even a pathological map cannot reintroduce the NaN."""
    bad = TuningMaps({"tau_mass": {"x": "pt", "centers": [25.0, 175.0], "values": [9.0, 20.0],
                                   "counts": [10, 10]}})
    out = ak.to_numpy(tau_visible_mass(_jets([45.0], [8.0], [1]), bad))
    assert out[0][0] < M_TAU


def test_empty_map_is_a_no_op_not_nan():
    empty = TuningMaps({"tau_mass": {"x": "pt", "centers": [], "values": [], "counts": []}})
    out = ak.to_numpy(tau_visible_mass(_jets([45.0], [8.0], [1]), empty))
    assert out[0][0] == pytest.approx(8.0)


def _leg(pt, eta, phi, m):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return {"px": np.array([px]), "py": np.array([py]), "pz": np.array([pz]),
            "e": np.array([np.sqrt(px * px + py * py + pz * pz + m * m)]),
            "mass": np.array([m]), "is_had": np.array([True])}


def test_fastmtt_recovers_events_the_jet_mass_destroyed():
    """The whole point: jet mass -> NaN, anchor mass -> a finite m_ττ."""
    met_x, met_y = np.array([25.0]), np.array([15.0])
    jet_mass = fastmtt_mass(_leg(60, 0.5, 0.0, 8.0), _leg(45, -0.8, 2.2, 11.0), met_x, met_y)
    anchor_mass = fastmtt_mass(_leg(60, 0.5, 0.0, 0.85), _leg(45, -0.8, 2.2, 0.85), met_x, met_y)
    assert np.isnan(jet_mass[0])                 # the bug
    assert np.isfinite(anchor_mass[0]) and anchor_mass[0] > 20.0   # the fix


def test_retag_jets_applies_tau_mass_last_so_escale_does_not_rescale_it():
    """tau_mass runs after escale: the assigned mass is final, not scaled again."""
    jets = _jets(pt=[45.0], mass=[8.0], tautag=[1])
    gen = ak.zip({"pid": _jag([15]), "pt": _jag([50.0]), "eta": _jag([0.3]),
                  "phi": _jag([0.0]), "mass": _jag([1.777]), "status": _jag([2]),
                  "m1": _jag([-1])})
    ev = SimpleNamespace(jets=jets, gen=gen)
    flat = lambda v: {"x": "pt", "centers": _CENTERS, "values": [v] * 4, "counts": [10] * 4}
    maps = TuningMaps({
        "tau_mass": flat(0.85),          # flat, so the lookup pT cannot confound the test
        "tau_eff": flat(1.0), "tau_mistag": flat(0.0),
        "bjet_escale": flat(1.0), "tau_escale": flat(0.5),
    })
    out, fields = retag_jets(ev, maps, np.random.default_rng(0))
    assert "tau_mass" in fields and "escale" in fields
    m = ak.to_numpy(out.mass)[0][0]
    # applied AFTER escale -> the map value stands (0.85). Applied before, the escale
    # would have scaled it to 0.425 and the τ mass would track the energy correction.
    assert m == pytest.approx(0.85, abs=1e-6), "escale must not rescale the assigned τ mass"


def test_tau_mass_absent_leaves_masses_alone():
    jets = _jets(pt=[45.0], mass=[8.0], tautag=[1])
    gen = ak.zip({"pid": _jag([15]), "pt": _jag([50.0]), "eta": _jag([0.3]),
                  "phi": _jag([0.0]), "mass": _jag([1.777]), "status": _jag([2]),
                  "m1": _jag([-1])})
    ev = SimpleNamespace(jets=jets, gen=gen)
    out, fields = retag_jets(ev, TuningMaps({}), np.random.default_rng(0))
    assert "tau_mass" not in fields
    assert ak.to_numpy(out.mass)[0][0] == pytest.approx(8.0)
