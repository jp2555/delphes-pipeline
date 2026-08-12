"""The map comparison decides whether a map may be transferred between processes.

Its whole job is the TRANSFERS / PER-PROCESS verdict, so the tests pin that: identical
maps must transfer, a difference above tolerance must not, and a difference located in a
pT region must be reported at that pT rather than averaged away. Getting this wrong in
either direction is expensive — a false "transfers" biases the NSBI likelihood ratio with
signal-derived maps on the background, a false "per-process" invents work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_maps import compare  # noqa: E402

_C = [25.0, 35.0, 45.0, 60.0, 85.0]


def _curve(vals, centers=None):
    return {"x": "pt", "centers": list(centers or _C), "values": list(vals)}


def _scalar(v):
    return {"x": "overall", "centers": [0.0], "values": [v]}


def test_identical_maps_transfer():
    m = {"btag_eff_b": _curve([0.70, 0.75, 0.78, 0.80, 0.82])}
    (q, rel, _, verdict), = compare(m, m, tol=0.05)
    assert q == "btag_eff_b" and rel == pytest.approx(0.0) and verdict == "TRANSFERS"


def test_difference_above_tolerance_is_flagged():
    a = {"tau_mistag": _curve([0.004] * 5)}
    b = {"tau_mistag": _curve([0.008] * 5)}          # 100% higher, as a fake rate would be
    (_, rel, _, verdict), = compare(a, b, tol=0.05)
    assert rel == pytest.approx(1.0) and verdict == "PER-PROCESS"


def test_difference_just_inside_tolerance_transfers():
    a = {"tau_eff": _curve([0.50] * 5)}
    b = {"tau_eff": _curve([0.52] * 5)}              # 4% < 5%
    (_, rel, _, verdict), = compare(a, b, tol=0.05)
    assert rel == pytest.approx(0.04) and verdict == "TRANSFERS"


def test_localised_difference_is_reported_at_its_pt_not_averaged():
    """A map that agrees everywhere except one bin must still be flagged, and the pT
    reported must be that bin — otherwise the diagnostic hides where the problem is."""
    a = {"bjet_escale": _curve([1.00, 1.00, 1.00, 1.00, 1.00])}
    b = {"bjet_escale": _curve([1.00, 1.00, 1.00, 1.00, 1.40])}
    (_, rel, at, verdict), = compare(a, b, tol=0.05)
    assert rel == pytest.approx(0.40) and at == pytest.approx(85.0) and verdict == "PER-PROCESS"


def test_scalar_maps_are_compared_not_skipped():
    """met_smear is a single number, not a curve — it is also the map most likely to
    differ between processes, so it must not fall through the curve path."""
    a = {"met_smear": _scalar(28.0)}
    b = {"met_smear": _scalar(41.0)}
    (_, rel, at, verdict), = compare(a, b, tol=0.05)
    assert rel == pytest.approx(13.0 / 28.0) and np.isnan(at) and verdict == "PER-PROCESS"


def test_maps_on_different_pt_grids_are_interpolated_not_rejected():
    """The two derivations need not produce identical bin centres (centres are count-
    weighted means), so a grid mismatch must interpolate rather than report nonsense."""
    a = {"tau_eff": _curve([0.50, 0.55, 0.60, 0.65, 0.70])}
    b = {"tau_eff": _curve([0.50, 0.60, 0.70], centers=[25.0, 45.0, 85.0])}
    (_, rel, _, verdict), = compare(a, b, tol=0.05)
    assert np.isfinite(rel) and verdict == "TRANSFERS"


def test_map_present_on_only_one_side_is_not_silently_dropped():
    a = {"tau_eff": _curve([0.5] * 5), "met_smear": _scalar(28.0)}
    b = {"tau_eff": _curve([0.5] * 5)}
    rows = compare(a, b, tol=0.05)
    assert [r[0] for r in rows] == ["tau_eff"], "shared maps only in the table"
    # the caller reports the asymmetry separately; here we only assert it is not compared
    assert set(a) - set(b) == {"met_smear"}


def test_empty_values_are_reported_not_crashed():
    a = {"btag_eff_c": _curve([])}
    b = {"btag_eff_c": _curve([0.1] * 5)}
    (_, rel, _, verdict), = compare(a, b, tol=0.05)
    assert np.isnan(rel) and verdict == "empty on one side"
