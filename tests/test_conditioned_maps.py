"""Conditioned maps — the v2 fix for the per-process detector model.

A map binned in pT alone is not a detector property, it is a per-process average.
Applying different such averages to signal and background writes the map DIFFERENCE
into the learned S/B ratio, which lands on the measured parameter. Conditioning on the
variable that EXPLAINS the process difference (jet true flavour, tau gen decay mode,
|eta|) is what makes one universal map applicable to every process.

v1 files are 1-D and must keep working unchanged.
"""
import numpy as np
import pytest

from delphes_pipeline.tuning.maps import TuningMaps, _lookup


def _1d(centers, values):
    return {"x": "pt", "centers": centers, "values": values,
            "counts": [100] * len(centers)}


def test_a_v1_one_dimensional_map_is_read_unchanged():
    m = _1d([20.0, 40.0], [0.5, 0.9])
    assert _lookup(m, [20.0, 30.0, 40.0]) == pytest.approx([0.5, 0.7, 0.9])


def test_flat_extrapolation_outside_the_grid_is_preserved():
    m = _1d([20.0, 40.0], [0.5, 0.9])
    assert _lookup(m, [1.0, 999.0]) == pytest.approx([0.5, 0.9])


def test_an_empty_grid_returns_the_default_not_zero():
    """A missing map must be a no-op for a multiplicative correction, not a zeroing."""
    assert _lookup({"centers": [], "values": []}, [50.0], default=1.0) == pytest.approx([1.0])


# --------------------------------------------------------------------------- #
# Categorical conditioning: jet true flavour, tau gen decay mode
# --------------------------------------------------------------------------- #
def _by_mode():
    return {"x": "gen_pt", "by": "decay_mode",
            "cats": {"0": _1d([20.0, 100.0], [0.80, 0.90]),      # 1-prong
                     "1": _1d([20.0, 100.0], [0.85, 0.95]),      # 1-prong + pi0
                     "10": _1d([20.0, 100.0], [0.70, 0.75])}}    # 3-prong


def test_each_category_reads_its_own_curve():
    got = _lookup(_by_mode(), [20.0, 20.0, 20.0], cat=[0, 1, 10])
    assert got == pytest.approx([0.80, 0.85, 0.70])


def test_categories_interpolate_independently_in_x():
    got = _lookup(_by_mode(), [60.0, 60.0], cat=[0, 10])
    assert got == pytest.approx([0.85, 0.725])


def test_an_unmapped_category_is_a_visible_no_op_not_a_borrowed_neighbour():
    """Silently borrowing a neighbouring mode is how a wrong response gets applied."""
    got = _lookup(_by_mode(), [20.0, 20.0], cat=[0, 11], default=1.0)
    assert got == pytest.approx([0.80, 1.0])


def test_an_explicit_catch_all_category_is_honoured():
    m = _by_mode()
    m["cats"]["_"] = _1d([20.0, 100.0], [0.5, 0.5])
    got = _lookup(m, [20.0, 20.0], cat=[0, 11])
    assert got == pytest.approx([0.80, 0.5])


def test_a_conditioned_map_read_without_a_category_raises():
    """Falling back to an average would be exactly the v1 defect, silently."""
    with pytest.raises(ValueError, match="conditioned on"):
        _lookup(_by_mode(), [50.0])


# --------------------------------------------------------------------------- #
# Second continuous axis: |eta| — C-3's fix (v1 applies a barrel SF everywhere)
# --------------------------------------------------------------------------- #
def _by_eta():
    return {"x": "pt", "y": "abs_eta", "y_edges": [0.0, 1.5, 2.5],
            "centers": [20.0, 100.0],
            "values": [[0.95, 0.98],        # barrel
                       [0.80, 0.88]],       # endcap
            "counts": [[10, 10], [10, 10]]}


def test_barrel_and_endcap_read_different_rows():
    got = _lookup(_by_eta(), [20.0, 20.0], y=[0.5, 2.0])
    assert got == pytest.approx([0.95, 0.80])


def test_eta_beyond_the_last_edge_uses_the_edge_row_not_the_default():
    got = _lookup(_by_eta(), [20.0], y=[4.0])
    assert got == pytest.approx([0.80])


def test_a_two_dimensional_map_read_without_y_raises():
    with pytest.raises(ValueError, match="second axis"):
        _lookup(_by_eta(), [50.0])


def test_categorical_and_eta_compose():
    """fake tau needs (jet flavour, pT, |eta|) — all three at once."""
    m = {"x": "pt", "by": "flavour",
         "cats": {"5": _by_eta(),
                  "0": {"x": "pt", "y": "abs_eta", "y_edges": [0.0, 1.5, 2.5],
                        "centers": [20.0, 100.0],
                        "values": [[0.10, 0.12], [0.20, 0.22]],
                        "counts": [[9, 9], [9, 9]]}}}
    got = _lookup(m, [20.0, 20.0, 20.0], cat=[5, 0, 0], y=[0.5, 0.5, 2.0])
    assert got == pytest.approx([0.95, 0.10, 0.20])


def test_the_public_api_exposes_the_conditioning():
    tm = TuningMaps({"tau_response": _by_mode()})
    got = tm.efficiency("tau_response", [20.0, 20.0], cat=[0, 10], default=1.0)
    assert got == pytest.approx([0.80, 0.70])
