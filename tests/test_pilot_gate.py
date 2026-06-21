"""Pilot Gate: passes on the good fixture, fails on the broken one.

The broken fixture (``broken=True``) injects three structural defects the Pilot
Gate is designed to catch: a >50% negative-weight fraction, all-zero
``Jet.Flavor`` (flavour association did not run), and no gen taus. Those three
checks must FAIL while the others are untouched here.
"""

from __future__ import annotations

from conftest import build_ctx

from delphes_pipeline.core.result import Severity
from delphes_pipeline.validation.pilot_gate import checks


def _by_name(results) -> dict:
    return {r.name: r for r in results}


def test_pilot_gate_all_pass_on_good(good_fixture_path):
    ctx = build_ctx(good_fixture_path)
    results = checks.run(ctx)

    gate = [r for r in results if r.severity is Severity.GATE]
    assert len(gate) == 6, f"expected six GATE checks, got {[r.name for r in gate]}"
    failed = [r.name for r in gate if not r.passed]
    assert not failed, f"good fixture should pass every Pilot-Gate check; failed: {failed}"


def test_pilot_gate_fails_on_broken(broken_fixture_path):
    ctx = build_ctx(broken_fixture_path)
    results = _by_name(checks.run(ctx))

    must_fail = (
        "pilot_gate.neg_weight_fraction",
        "pilot_gate.jet_flavor_filled",
        "pilot_gate.gen_taus_present",
    )
    for name in must_fail:
        assert name in results, f"missing check {name}"
        assert not results[name].passed, f"{name} should FAIL on the broken fixture"
        assert results[name].severity is Severity.GATE
