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
def _prof(centers, med, q95_over_med, n=20000):
    """A quantile profile with a prescribed median, q95/median and per-bin count."""
    lv = _LEVELS
    rows = []
    for m, sc in zip(med, q95_over_med):
        # piecewise-linear in level: median at 0.5, q95 at 0.95
        rows.append([float(np.interp(x, [0.0, 0.5, 0.95, 1.0],
                                     [m * 0.8, m, m * sc, m * sc * 1.05])) for x in lv])
    return SimpleNamespace(centers=np.asarray(centers, dtype=float),
                           counts=np.full(len(centers), n, dtype=int),
                           aux={"quantile_levels": lv.tolist(), "quantile_values": rows})


def test_transfer_passes_when_two_anchors_agree():
    import check_anchor_transfer as T

    a = T._rows(_prof([25.0, 45.0], [0.99, 0.98], [1.13, 1.12]))
    b = T._rows(_prof([25.0, 45.0], [0.992, 0.978], [1.14, 1.12]))
    assert all(v[-1] for v in T.compare(a, b))


def test_transfer_fails_on_a_scale_shift_because_it_enters_mtautau_linearly():
    import check_anchor_transfer as T

    a = T._rows(_prof([25.0], [0.99], [1.13]))
    b = T._rows(_prof([25.0], [1.04], [1.13]))            # 5% median shift, high stats
    (_, _, _, _, d_med, z_med, _, _, ok), = T.compare(a, b)
    assert not ok and d_med > 0.01 and z_med > 2.0


def test_transfer_fails_on_a_shape_change_even_at_identical_median():
    """The case option B is about: same scale, different tail."""
    import check_anchor_transfer as T

    a = T._rows(_prof([25.0], [0.99], [1.13]))
    b = T._rows(_prof([25.0], [0.99], [1.45]))
    (_, _, _, _, d_med, _, d_shape, z_shape, ok), = T.compare(a, b)
    assert not ok and d_med < 0.01 and d_shape > 0.05 and z_shape > 2.0


# --------------------------------------------------------------------------- #
# the statistical floor: a sparse bin must not be called a process difference
# --------------------------------------------------------------------------- #
def test_a_sparse_bin_is_not_flagged_even_when_the_shape_differs_a_lot():
    """Fake τ are rare, so q95 from a few hundred entries carries several percent of
    noise on its own. Without a floor the tolerance alone manufactures 'DIFFERS'."""
    import check_anchor_transfer as T

    a = T._rows(_prof([25.0], [0.99], [1.13], n=40))
    b = T._rows(_prof([25.0], [0.99], [1.195], n=40))     # ~6% apart, on 40 entries
    (_, n, _, _, _, _, d_shape, z_shape, ok), = T.compare(a, b)
    assert n == 40 and d_shape > 0.05, "the raw difference does exceed the tolerance"
    assert z_shape < 2.0 and ok, "but it is not significant, so it must not be flagged"


def test_the_same_difference_IS_flagged_once_the_statistics_support_it():
    """Same shape difference, 200x the events -> now a real process difference."""
    import check_anchor_transfer as T

    a = T._rows(_prof([25.0], [0.99], [1.13], n=30000))
    b = T._rows(_prof([25.0], [0.99], [1.195], n=30000))   # the SAME 6%, 750x the events
    (_, _, _, _, _, _, _, z_shape, ok), = T.compare(a, b)
    assert not ok and z_shape > 2.0


def test_counts_are_reported_so_a_verdict_can_be_judged():
    import check_anchor_transfer as T

    a = T._rows(_prof([25.0], [0.99], [1.13], n=1234))
    assert a[0][1] == 1234


# --------------------------------------------------------------------------- #
# thin-bin guard: a quantile grid needs far more entries than a mean, and
# min_bin_count gates only the closure verdict, never the derivation
# --------------------------------------------------------------------------- #
def _two_bin_map(counts):
    row = lambda v: [v] * _RESPONSE_QUANTILES
    return {"x": "pt", "centers": [25.0, 75.0], "counts": list(counts),
            "quantile_levels": _LEVELS.tolist(),
            "quantile_values": [row(1.0), row(5.0)]}     # bin 1 is absurd, as a thin bin is


def test_a_thin_quantile_bin_is_not_sampled():
    """The tt̄ anchor produced a median τ mass of 0.14 = m_π in its top pT bin — a
    one-prong artefact of too few entries. Sampling such a row applies it to every
    object in that pT range."""
    from delphes_pipeline.tuning.maps import MIN_QUANTILE_COUNT

    ev = _events(np.tile([25.0, 75.0], _N // 2))
    thin = _two_bin_map([5000, MIN_QUANTILE_COUNT // 4])
    jets, _, _ = retag_jets(ev, TuningMaps({"tau_response": thin}), np.random.default_rng(0))
    pt = ak.to_numpy(ak.flatten(jets.pt))
    assert pt[0::2] == pytest.approx(25.0 * 1.0)
    assert pt[1::2] == pytest.approx(75.0 * 1.0), "thin bin must borrow the populated row"


def test_a_populated_bin_is_used_normally():
    from delphes_pipeline.tuning.maps import MIN_QUANTILE_COUNT

    ev = _events(np.tile([25.0, 75.0], _N // 2))
    ok = _two_bin_map([5000, MIN_QUANTILE_COUNT * 3])
    jets, _, _ = retag_jets(ev, TuningMaps({"tau_response": ok}), np.random.default_rng(0))
    pt = ak.to_numpy(ak.flatten(jets.pt))
    assert pt[1::2] == pytest.approx(75.0 * 5.0), "a populated bin keeps its own row"


def test_a_map_with_no_counts_is_trusted_as_written():
    """Older maps files carry no counts; they must still work."""
    ev = _events(np.tile([25.0, 75.0], _N // 2))
    m = _two_bin_map([5000, 10])
    del m["counts"]
    jets, _, _ = retag_jets(ev, TuningMaps({"tau_response": m}), np.random.default_rng(0))
    assert ak.to_numpy(ak.flatten(jets.pt))[1::2] == pytest.approx(75.0 * 5.0)


def test_counts_centers_and_quantile_rows_stay_index_aligned():
    """The guard reads counts to judge a quantile row, so a desync would silently
    redirect the wrong bins. Both are built by the same per-bin loop over the anchor."""
    import delphes_pipeline.tuning.anchor as A

    lv = _RESPONSE_LEVELS
    x = np.concatenate([np.full(900, 25.0), np.full(900, 45.0)])   # bin 30-40 left empty
    vals = np.concatenate([np.full(900, 1.0), np.full(900, 2.0)])
    prof = __import__("delphes_pipeline.core.observables", fromlist=["x"]).binned_response(
        x, vals, [20, 30, 40, 50], quantity="t", x="pt")
    qv = []
    for lo, hi in zip([20, 30, 40], [30, 40, 50]):
        m = (x >= lo) & (x < hi)
        if not m.any():
            continue
        qv.append(np.quantile(vals[m], lv).tolist())
    assert len(qv) == len(prof.centers) == len(prof.counts)
