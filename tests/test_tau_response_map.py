"""Option B: the τ energy response is SAMPLED from the anchor, not scaled to its median.

The whole point is the property a multiplicative map cannot have. ``tau_escale`` aligns
medians and leaves the shape alone; measured, the Delphes response is one-sidedly broader
at matched median (3× at 20--30 GeV), so its excess survives into m_vis above the
kinematic limit — a scale factor cannot depopulate a forbidden region.

So the tests here assert the SHAPE, not the mean: after resampling, the reconstructed
response distribution must match the anchor's including its tail, starting from a Delphes
response that is deliberately far too broad. Plus the three things that make it usable at
production scale: fakes get their own reference (they have no gen τ), MET follows the
energy change, and a maps file without quantiles still takes the legacy escale path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from delphes_pipeline.tuning.anchor import _RESPONSE_LEVELS, _RESPONSE_QUANTILES  # noqa: E402
from delphes_pipeline.tuning.maps import (  # noqa: E402
    RetaggedEvents, TuningMaps, propagate_to_met, resample_tau_energy, retag_jets,
)

_N = 4000
# track the production grid: a coarser one inflates the tail by interpolating from the last
# stored level to the sample max, which is exactly what this suite exists to catch
_LEVELS = _RESPONSE_LEVELS
_CENTERS = [25.0, 40.0, 75.0]


def _resp_map(sample_fn):
    """A quantile map built from a target response distribution, one row per pT bin."""
    rng = np.random.default_rng(0)
    return {"x": "pt", "centers": list(_CENTERS),
            "counts": [10000] * len(_CENTERS),
            "quantile_levels": _LEVELS.tolist(),
            "quantile_values": [np.quantile(sample_fn(rng, 20000), _LEVELS).tolist()
                                for _ in _CENTERS]}


def _cms_like(rng, n):
    """Narrow, mildly asymmetric — the anchor: median ~1, q95/med ~1.13."""
    return np.exp(rng.normal(0.0, 0.06, n))


def _delphes_like(rng, n):
    """Same median, one-sided high tail — what the escale leaves behind."""
    x = np.exp(rng.normal(0.0, 0.06, n))
    k = rng.random(n) < 0.15
    x[k] *= rng.uniform(1.5, 3.0, k.sum())
    return x


def _jag(vals):
    return ak.Array([[v] for v in vals])


def _events(gen_pt, *, with_genjet=False, met=40.0, phi=0.3):
    """One τ-jet per event, sitting on one visible gen τ (collinear, so it always matches)."""
    eta = np.zeros(_N)
    jphi = np.linspace(-3.0, 3.0, _N)
    jets = ak.zip({"pt": _jag(gen_pt * 1.4), "eta": _jag(eta), "phi": _jag(jphi),
                   "mass": _jag(np.full(_N, 2.0)), "btag": _jag(np.zeros(_N)),
                   "tautag": _jag(np.ones(_N)), "flavor": _jag(np.full(_N, 15.0)),
                   "charge": _jag(np.zeros(_N))})
    # gen record: a hadronic τ plus its ν, so gen_visible_taus reconstructs pT = gen_pt
    vis = gen_pt
    full = vis / 0.65
    gen = ak.zip({
        "pid": ak.Array([[15, 16]] * _N), "status": ak.Array([[2, 1]] * _N),
        "m1": ak.Array([[-1, 0]] * _N),
        "pt": ak.Array([[f, f - v] for f, v in zip(full, vis)]),
        "eta": ak.Array([[e, e] for e in eta]), "phi": ak.Array([[p, p] for p in jphi]),
        "mass": ak.Array([[1.777, 0.0]] * _N)})
    ev = SimpleNamespace(
        jets=jets, gen=gen, n=_N,
        met=ak.zip({"met": ak.Array(np.full(_N, met)), "eta": ak.Array(np.zeros(_N)),
                    "phi": ak.Array(np.full(_N, phi))}))
    if with_genjet:
        ev.genjets = ak.zip({"pt": _jag(gen_pt), "eta": _jag(eta), "phi": _jag(jphi),
                             "mass": _jag(np.zeros(_N))})
    else:
        ev.genjets = ak.Array([[] for _ in range(_N)])
    return ev


def _reconstructed_response(ev, maps, seed=0):
    jets, fields, _ = retag_jets(ev, maps, np.random.default_rng(seed))
    reco = ak.to_numpy(ak.flatten(jets.pt))
    gen = ak.to_numpy(ak.flatten(ev.gen.pt))[0::2] * 0.65      # visible τ pT
    return reco / gen, fields


# --------------------------------------------------------------------------- #
# the property a multiplicative map cannot have
# --------------------------------------------------------------------------- #
def test_resampling_reproduces_the_anchor_tail_not_just_the_median():
    """Starting from a Delphes response 15% contaminated with a 1.5-3x tail, resampling
    must land on the CMS SHAPE — the upper quantiles, which an escale cannot move."""
    gen_pt = np.full(_N, 40.0)
    ev = _events(gen_pt)
    maps = TuningMaps({"tau_response": _resp_map(_cms_like)})
    got, fields = _reconstructed_response(ev, maps)
    assert "tau_response" in fields
    target = _cms_like(np.random.default_rng(7), 200000)
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        assert np.quantile(got, q) == pytest.approx(np.quantile(target, q), rel=0.06), q


def test_median_alignment_alone_would_leave_the_tail():
    """Control: the escale path (median ratio) cannot remove a one-sided tail, which is
    why option B exists. Aligning medians leaves q95 far above the anchor's."""
    rng = np.random.default_rng(3)
    delphes, cms = _delphes_like(rng, 200000), _cms_like(rng, 200000)
    aligned = delphes * (np.median(cms) / np.median(delphes))
    assert np.quantile(aligned, 0.95) > np.quantile(cms, 0.95) * 1.3


def test_resampling_is_binned_in_gen_pt_so_each_bin_gets_its_own_row():
    """The map is looked up at the GEN pT it was binned in — no reco-pT Newton step."""
    gen_pt = np.tile([25.0, 75.0], _N // 2)
    ev = _events(gen_pt)
    hi = {"x": "pt", "centers": [25.0, 75.0], "counts": [1000, 1000],
          "quantile_levels": _LEVELS.tolist(),
          "quantile_values": [[0.5] * _RESPONSE_QUANTILES, [2.0] * _RESPONSE_QUANTILES]}       # bin 0 -> 0.5x, bin 1 -> 2x
    jets, _, _ = retag_jets(ev, TuningMaps({"tau_response": hi}), np.random.default_rng(0))
    pt = ak.to_numpy(ak.flatten(jets.pt))
    assert pt[0::2] == pytest.approx(25.0 * 0.5)
    assert pt[1::2] == pytest.approx(75.0 * 2.0)


# --------------------------------------------------------------------------- #
# fakes: TTto4Q selects them exclusively, and they have no gen tau
# --------------------------------------------------------------------------- #
def test_fake_taus_use_the_genjet_reference_when_no_gen_tau_exists():
    gen_pt = np.full(_N, 40.0)
    ev = _events(gen_pt, with_genjet=True)
    ev.gen = ak.zip({"pid": ak.Array([[21]] * _N), "status": ak.Array([[1]] * _N),
                     "m1": ak.Array([[-1]] * _N), "pt": _jag(np.full(_N, 50.0)),
                     "eta": _jag(np.full(_N, 4.0)), "phi": _jag(np.zeros(_N)),
                     "mass": _jag(np.zeros(_N))})          # no gen τ anywhere near
    fake = {"x": "pt", "centers": [40.0], "counts": [1000],
            "quantile_levels": _LEVELS.tolist(), "quantile_values": [[0.6] * _RESPONSE_QUANTILES]}
    real = {"x": "pt", "centers": [40.0], "counts": [1000],
            "quantile_levels": _LEVELS.tolist(), "quantile_values": [[1.0] * _RESPONSE_QUANTILES]}
    jets, fields, _ = retag_jets(ev, TuningMaps({"tau_response": real,
                                                 "tau_fake_response": fake}),
                                 np.random.default_rng(0))
    assert "tau_response" in fields
    assert ak.to_numpy(ak.flatten(jets.pt)) == pytest.approx(40.0 * 0.6)


def test_real_tau_match_takes_precedence_over_the_fake_map():
    gen_pt = np.full(_N, 40.0)
    ev = _events(gen_pt, with_genjet=True)
    maps = TuningMaps({
        "tau_response": {"x": "pt", "centers": [40.0], "counts": [1000],
                         "quantile_levels": _LEVELS.tolist(),
                         "quantile_values": [[1.0] * _RESPONSE_QUANTILES]},
        "tau_fake_response": {"x": "pt", "centers": [40.0], "counts": [1000],
                              "quantile_levels": _LEVELS.tolist(),
                              "quantile_values": [[9.0] * _RESPONSE_QUANTILES]}})
    jets, _, _ = retag_jets(ev, maps, np.random.default_rng(0))
    assert ak.to_numpy(ak.flatten(jets.pt)) == pytest.approx(40.0)   # not 360


def test_jet_with_neither_reference_is_left_untouched():
    ev = _events(np.full(_N, 40.0))          # no genjets, and we give no tau_response
    fake = {"x": "pt", "centers": [40.0], "counts": [1000],
            "quantile_levels": _LEVELS.tolist(), "quantile_values": [[0.1] * _RESPONSE_QUANTILES]}
    jets, _, _ = retag_jets(ev, TuningMaps({"tau_fake_response": fake}),
                            np.random.default_rng(0))
    assert ak.to_numpy(ak.flatten(jets.pt)) == pytest.approx(40.0 * 1.4)   # unchanged


# --------------------------------------------------------------------------- #
# MET must follow the energy change, since FastMTT fits against it
# --------------------------------------------------------------------------- #
def test_met_absorbs_the_tau_energy_change_vectorially():
    before = ak.zip({"pt": _jag(np.full(3, 50.0)), "phi": _jag(np.zeros(3))})
    after = ak.zip({"pt": _jag(np.full(3, 30.0)), "phi": _jag(np.zeros(3))})
    met = ak.zip({"met": ak.Array(np.full(3, 10.0)), "eta": ak.Array(np.zeros(3)),
                  "phi": ak.Array(np.zeros(3))})
    out = propagate_to_met(met, before, after)
    # the τ lost 20 GeV along +x, so the recoil must gain it: 10 - (30-50) = 30
    assert ak.to_numpy(out.met) == pytest.approx(30.0)
    assert ak.to_numpy(out.phi) == pytest.approx(0.0)


def test_retag_returns_a_propagated_met_only_when_resampling():
    ev = _events(np.full(_N, 40.0))
    _, _, met = retag_jets(ev, TuningMaps({"tau_response": _resp_map(_cms_like)}),
                           np.random.default_rng(0), propagate_met=True)
    assert met is not None
    _, _, off = retag_jets(ev, TuningMaps({"tau_response": _resp_map(_cms_like)}),
                           np.random.default_rng(0))
    assert off is None, "MET propagation must be opt-in"
    _, _, met_legacy = retag_jets(ev, TuningMaps({}), np.random.default_rng(0))
    assert met_legacy is None


def test_retagged_events_smears_on_top_of_the_propagated_met():
    """If smear_met ran on the RAW met the propagation would be silently discarded."""
    ev = _events(np.full(_N, 40.0), met=40.0)
    maps = TuningMaps({"tau_response": {"x": "pt", "centers": [40.0], "counts": [1000],
                                        "quantile_levels": _LEVELS.tolist(),
                                        "quantile_values": [[0.5] * _RESPONSE_QUANTILES]},
                       "met_smear": {"x": "overall", "centers": [0.0], "values": [1e-6]}})
    r = RetaggedEvents(ev, maps, np.random.default_rng(0), propagate_met=True)
    assert {"tau_response", "met_smear"} <= r.retagged_fields
    # τ went 56 -> 20 GeV along phi=jet; MET must have moved well away from its raw 40
    assert abs(float(np.mean(ak.to_numpy(r.met.met))) - 40.0) > 1.0


# --------------------------------------------------------------------------- #
# backward compatibility: a maps file without quantiles keeps the escale path
# --------------------------------------------------------------------------- #
def test_maps_without_quantiles_fall_back_to_the_multiplicative_escale():
    ev = _events(np.full(_N, 40.0))
    legacy = TuningMaps({"bjet_escale": {"x": "pt", "centers": [40.0], "values": [1.0]},
                         "tau_escale": {"x": "pt", "centers": [40.0], "values": [0.5]}})
    jets, fields, met = retag_jets(ev, legacy, np.random.default_rng(0))
    assert "tau_response" not in fields and "escale" in fields
    assert met is None
    assert ak.to_numpy(ak.flatten(jets.pt)) == pytest.approx(56.0 * 0.5, rel=1e-6)


# --------------------------------------------------------------------------- #
# the transfer criterion: one response map per campaign, or one per process?
# --------------------------------------------------------------------------- #
def _prof(centers, med, q95_over_med):
    """A quantile profile with a prescribed median and q95/median per bin."""
    lv = np.linspace(0.0, 1.0, _RESPONSE_QUANTILES)
    rows = []
    for m, s in zip(med, q95_over_med):
        # piecewise-linear in level: median at 0.5, q95 at 0.95
        rows.append([float(np.interp(x, [0.0, 0.5, 0.95, 1.0],
                                     [m * 0.8, m, m * s, m * s * 1.05])) for x in lv])
    return SimpleNamespace(centers=np.asarray(centers, dtype=float),
                           aux={"quantile_levels": lv.tolist(), "quantile_values": rows})


def test_transfer_passes_when_two_anchors_agree():
    import check_anchor_transfer as T

    a = T._rows(_prof([25.0, 45.0], [0.99, 0.98], [1.13, 1.12]))
    b = T._rows(_prof([25.0, 45.0], [0.992, 0.978], [1.14, 1.12]))
    assert all(ok for *_, ok in T.compare(a, b))


def test_transfer_fails_on_a_scale_shift_because_it_enters_mtautau_linearly():
    import check_anchor_transfer as T

    a = T._rows(_prof([25.0], [0.99], [1.13]))
    b = T._rows(_prof([25.0], [1.04], [1.13]))            # 5% median shift
    (_, _, _, d_med, _, ok), = T.compare(a, b)
    assert not ok and d_med > 0.01


def test_transfer_fails_on_a_shape_change_even_at_identical_median():
    """The case option B is about: same scale, different tail."""
    import check_anchor_transfer as T

    a = T._rows(_prof([25.0], [0.99], [1.13]))
    b = T._rows(_prof([25.0], [0.99], [1.45]))
    (_, _, _, d_med, d_shape, ok), = T.compare(a, b)
    assert not ok and d_med < 0.01 and d_shape > 0.05


# --------------------------------------------------------------------------- #
# SCOPE. Resampling REPLACES the pT rather than scaling it, so anything it reaches
# loses both its reconstructed energy and its own escale. Nearly every jet has a
# GenJet within dR<0.4, so an ungated fake branch rewrites the whole event: an
# earlier version did exactly that and moved an m_bb proxy by -24%. These tests use
# multi-jet events on purpose — a one-jet fixture cannot see it.
# --------------------------------------------------------------------------- #
def _multijet():
    """2 b-jets + 1 τ-jet + 1 light jet per event, every jet with a GenJet, no gen τ."""
    n = 600
    rows = [(100.0, 0.5, 0.0, 5, 0), (80.0, -0.6, 1.2, 5, 0),      # b-jets
            (45.0, 0.2, 2.4, 15, 1), (60.0, 1.1, -2.0, 0, 0)]      # τ-jet, light jet
    col = lambda i: ak.Array([[r[i] for r in rows]] * n)
    jets = ak.zip({"pt": col(0), "eta": col(1), "phi": col(2),
                   "mass": ak.Array([[12.0, 10.0, 2.0, 8.0]] * n),
                   "flavor": col(3), "tautag": col(4),
                   "btag": ak.Array([[1, 1, 0, 0]] * n),
                   "charge": ak.Array([[0.0] * 4] * n)})
    genjets = ak.zip({"pt": col(0), "eta": col(1), "phi": col(2),
                      "mass": ak.Array([[0.0] * 4] * n)})
    gen = ak.zip({"pid": ak.Array([[21]] * n), "status": ak.Array([[1]] * n),
                  "m1": ak.Array([[-1]] * n), "pt": ak.Array([[50.0]] * n),
                  "eta": ak.Array([[6.0]] * n), "phi": ak.Array([[0.0]] * n),
                  "mass": ak.Array([[0.0]] * n)})
    return SimpleNamespace(jets=jets, gen=gen, genjets=genjets, n=n,
                           met=ak.zip({"met": ak.Array(np.full(n, 30.0)),
                                       "eta": ak.Array(np.zeros(n)),
                                       "phi": ak.Array(np.zeros(n))}))


def _flat(a):
    return ak.to_numpy(ak.flatten(a))


def _both_maps(real=1.0, fake=0.6):
    row = lambda v: [v] * _RESPONSE_QUANTILES
    mk = lambda v: {"x": "pt", "centers": [45.0, 100.0], "counts": [1000, 1000],
                    "quantile_levels": _LEVELS.tolist(),
                    "quantile_values": [row(v), row(v)]}
    return TuningMaps({"tau_response": mk(real), "tau_fake_response": mk(fake)})


def test_fake_resampling_never_touches_b_jets_or_light_jets():
    ev = _multijet()
    jets, _, _ = retag_jets(ev, _both_maps(), np.random.default_rng(0))
    pt = _flat(jets.pt).reshape(-1, 4)
    assert pt[:, 0] == pytest.approx(100.0), "b-jet 1 must be untouched"
    assert pt[:, 1] == pytest.approx(80.0), "b-jet 2 must be untouched"
    assert pt[:, 3] == pytest.approx(60.0), "light jet must be untouched"
    assert pt[:, 2] == pytest.approx(45.0 * 0.6), "only the τ candidate is resampled"


def test_mbb_is_invariant_under_resampling():
    """m_bb already agrees with CMS; resampling must not be able to move it."""
    ev = _multijet()
    jets, _, _ = retag_jets(ev, _both_maps(fake=0.4), np.random.default_rng(0))
    pt = _flat(jets.pt).reshape(-1, 4)
    assert np.allclose(pt[:, :2], np.array([100.0, 80.0]))


def test_bjet_escale_survives_resampling():
    """An ungated resample overwrote the pT outright, silently discarding bjet_escale."""
    ev = _multijet()
    m = _both_maps()
    m.maps["bjet_escale"] = {"x": "pt", "centers": [90.0], "values": [1.2]}
    m.maps["tau_escale"] = {"x": "pt", "centers": [45.0], "values": [1.0]}
    jets, fields, _ = retag_jets(ev, m, np.random.default_rng(0))
    pt = _flat(jets.pt).reshape(-1, 4)
    assert "escale" in fields
    assert pt[:, 0] == pytest.approx(120.0, rel=1e-6), "b-jet keeps its own escale"


def test_tau_candidates_outside_the_anchor_acceptance_are_left_alone():
    """The map was measured on pT>20, |eta|<2.5; applying it outside is extrapolation."""
    n = 200
    jets = ak.zip({"pt": ak.Array([[15.0, 45.0]] * n), "eta": ak.Array([[0.2, 3.1]] * n),
                   "phi": ak.Array([[0.0, 1.0]] * n), "mass": ak.Array([[2.0, 2.0]] * n),
                   "flavor": ak.Array([[15, 15]] * n), "tautag": ak.Array([[1, 1]] * n),
                   "btag": ak.Array([[0, 0]] * n), "charge": ak.Array([[0.0, 0.0]] * n)})
    ev = SimpleNamespace(jets=jets, n=n,
                         genjets=ak.zip({"pt": ak.Array([[15.0, 45.0]] * n),
                                         "eta": ak.Array([[0.2, 3.1]] * n),
                                         "phi": ak.Array([[0.0, 1.0]] * n),
                                         "mass": ak.Array([[0.0, 0.0]] * n)}),
                         gen=ak.zip({"pid": ak.Array([[21]] * n), "status": ak.Array([[1]] * n),
                                     "m1": ak.Array([[-1]] * n), "pt": ak.Array([[9.0]] * n),
                                     "eta": ak.Array([[7.0]] * n), "phi": ak.Array([[0.0]] * n),
                                     "mass": ak.Array([[0.0]] * n)}),
                         met=ak.zip({"met": ak.Array(np.full(n, 20.0)),
                                     "eta": ak.Array(np.zeros(n)),
                                     "phi": ak.Array(np.zeros(n))}))
    jets_out, _, _ = retag_jets(ev, _both_maps(fake=0.5), np.random.default_rng(0))
    pt = _flat(jets_out.pt).reshape(-1, 2)
    assert pt[:, 0] == pytest.approx(15.0), "below the 20 GeV floor"
    assert pt[:, 1] == pytest.approx(45.0), "beyond |eta| 2.5"


def test_fake_map_alone_does_not_enable_resampling():
    """A half-populated maps file must fall back, not route real τ through the fake map."""
    ev = _multijet()
    only_fake = TuningMaps({"tau_fake_response": _both_maps().maps["tau_fake_response"]})
    jets, fields, _ = retag_jets(ev, only_fake, np.random.default_rng(0))
    assert "tau_response" not in fields
    assert _flat(jets.pt).reshape(-1, 4)[:, 2] == pytest.approx(45.0)
