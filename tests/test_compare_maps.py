"""Map universality must be judged within statistics, not by a flat relative cut.

E3's pass condition is that per-process determinations agree in common bins. The
failing bins in the real comparison are the highest-pT ones, which are also the
emptiest -- so a relative cut alone cannot separate "the parameterisation is
missing a variable" from "this bin has 468 entries".
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_maps  # noqa: E402


def _m(values, counts, **kw):
    d = {"x": "pt", "centers": [25.0, 35.0, 45.0], "values": values, "counts": counts}
    d.update(kw)
    return d


def _rows(a, b, **kw):
    return {r[0]: r for r in compare_maps.compare(a, b, tol=0.05, **kw)}


def test_a_big_relative_difference_in_an_empty_bin_is_not_called_inconsistent():
    a = {"e": _m([0.40, 0.60, 0.70], [30000, 20000, 12])}
    b = {"e": _m([0.40, 0.60, 0.45], [30000, 20000, 12])}
    q, rel, at, pull, verdict = _rows(a, b)["e"]
    assert rel > 0.30, "the relative difference is large"
    assert pull < 3.0 and verdict == "consistent", "but 12 entries cannot resolve it"


def test_a_small_relative_difference_with_huge_statistics_is_called_inconsistent():
    a = {"e": _m([0.400, 0.60, 0.70], [10_000_000, 20000, 5000])}
    b = {"e": _m([0.412, 0.60, 0.70], [10_000_000, 20000, 5000])}
    q, rel, at, pull, verdict = _rows(a, b)["e"]
    assert rel < 0.05, "under the flat tolerance"
    assert pull > 3.0 and verdict == "DIFFERS", "yet far outside the errors"


def test_identical_maps_are_consistent():
    a = {"e": _m([0.4, 0.6, 0.7], [1000, 1000, 1000])}
    assert _rows(a, dict(a))["e"][4] == "consistent"


def test_a_verdict_without_a_usable_error_is_flagged_with_a_star():
    """escale/SF maps carry no error; the verdict must not read as statistical."""
    a = {"s": _m([1.4, 1.5, 1.6], [])}
    b = {"s": _m([1.4, 1.5, 2.6], [])}
    assert _rows(a, b)["s"][4] == "DIFFERS*"


def test_binomial_error_is_used_for_efficiency_maps():
    s = compare_maps._sigma(_m([0.5], [10000]))
    assert s[0] == pytest.approx(np.sqrt(0.25 / 10000), rel=1e-6)


def test_a_width_map_uses_the_width_error_not_the_binomial_one():
    """met_resolution values are GeV, not probabilities."""
    s = compare_maps._sigma({"x": "ht", "centers": [100.0], "values": [30.0],
                             "counts": [5000]})
    assert s[0] == pytest.approx(30.0 / np.sqrt(2 * 5000), rel=1e-6)


def test_the_prescription_is_a_missing_variable_not_a_per_process_patch(capsys):
    a = {"e": _m([0.40, 0.60, 0.70], [10_000_000] * 3)}
    b = {"e": _m([0.55, 0.60, 0.70], [10_000_000] * 3)}
    with tempfile.TemporaryDirectory() as td:
        fa, fb = Path(td) / "a.json", Path(td) / "b.json"
        fa.write_text(json.dumps({"maps": a}))
        fb.write_text(json.dumps({"maps": b}))
        compare_maps.main([str(fa), str(fb), "--out", td])
    said = capsys.readouterr().out
    assert "missing" in said.lower() and "VARIABLE" in said
    assert "per-process" not in said.lower().replace("per process.", "")
