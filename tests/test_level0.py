"""Level-0 object response on the card-injected good fixture.

Two assertions per measured quantity:

1. Every Level-0 ``CheckResult`` of ``GATE`` severity passes (closure against the
   card holds on a fixture built *from* the card).
2. For each efficiency/rate quantity, the per-bin measured value in
   ``result.extra`` recovers the injected truth — i.e. ``card_formulas.expected``
   at the bin center (evaluated at eta=0, as the leaf modules do) — within
   ~3 binomial sigma or an absolute 0.05, whichever is looser.

MET has no card formula, so it is asserted as an overall resolution near the
injected ~20 GeV rather than a closure.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import build_ctx

from delphes_pipeline.core.result import Severity
from delphes_pipeline.validation.level0_objects import btag, leptons, met, tau
from delphes_pipeline.validation.references import card_formulas


def _gate_passes(results):
    """All GATE-severity results passed."""
    return [r.name for r in results if r.severity is Severity.GATE and not r.passed]


_MIN_BIN_COUNT = 25  # mirror config.example.yml level0.min_bin_count


def _recovers_truth(quantity: str, extra: dict):
    """Assert per-bin measured recovers the card formula within 3σ or 0.05.

    Only bins the modules actually closure-test (``count >= min_bin_count``) are
    asserted; sparse high-pT tail bins are dominated by edge statistics (e.g. a
    handful of leptons all matched -> measured 1.0 with degenerate err 0) and are
    not part of the closure. The statistical band uses the *null* binomial error
    (from the card-expected rate and the bin count), so it never collapses to
    zero at a measured rate of 0 or 1. The leaf modules evaluate the closure
    target at eta=0, so we compare against ``card_formulas.expected`` there.
    """
    centers = np.asarray(extra["centers"], dtype=float)
    measured = np.asarray(extra["measured"], dtype=float)
    counts = np.asarray(extra["counts"], dtype=int)
    assert centers.size > 0, f"{quantity}: no populated bins"

    tested = counts >= _MIN_BIN_COUNT
    assert tested.any(), f"{quantity}: no bin reaches min_bin_count"

    expected = card_formulas.expected(quantity, centers, np.zeros_like(centers))
    null_err = np.sqrt(np.maximum(expected * (1.0 - expected), 0.0) / np.maximum(counts, 1))
    tol = np.maximum(3.0 * null_err, 0.05)
    bad = tested & (np.abs(measured - expected) > tol)
    assert not bad.any(), (
        f"{quantity}: bins not recovering truth at "
        f"{centers[bad].tolist()} GeV: measured {measured[bad].tolist()} "
        f"vs expected {expected[bad].tolist()} (tol {tol[bad].tolist()})"
    )


@pytest.fixture(scope="module")
def ctx(good_fixture_path):
    return build_ctx(good_fixture_path)


# --------------------------------------------------------------------------- #
# b-tag: three quantities (b eff, c eff, light mistag)
# --------------------------------------------------------------------------- #
def test_btag_gate_passes(ctx):
    results = btag.run(ctx)
    assert not _gate_passes(results)


@pytest.mark.parametrize(
    "quantity", ["btag_eff_b", "btag_eff_c", "btag_mistag_light"]
)
def test_btag_recovers_truth(ctx, quantity):
    results = {r.extra["quantity"]: r for r in btag.run(ctx)}
    _recovers_truth(quantity, results[quantity].extra)


# --------------------------------------------------------------------------- #
# tau: efficiency + mistag
# --------------------------------------------------------------------------- #
def test_tau_gate_passes(ctx):
    results = tau.run(ctx)
    assert not _gate_passes(results)


@pytest.mark.parametrize(
    "name,quantity",
    [("level0.tau.tau_eff", "tau_eff"), ("level0.tau.tau_mistag", "tau_mistag")],
)
def test_tau_recovers_truth(ctx, name, quantity):
    results = {r.name: r for r in tau.run(ctx)}
    _recovers_truth(quantity, results[name].extra)


# --------------------------------------------------------------------------- #
# leptons: electron + muon efficiency
# --------------------------------------------------------------------------- #
def test_leptons_gate_passes(ctx):
    results = leptons.run(ctx)
    assert not _gate_passes(results)


@pytest.mark.parametrize(
    "name,quantity",
    [
        ("level0.leptons.electron_eff", "electron_eff"),
        ("level0.leptons.muon_eff", "muon_eff"),
    ],
)
def test_leptons_recovers_truth(ctx, name, quantity):
    results = {r.name: r for r in leptons.run(ctx)}
    _recovers_truth(quantity, results[name].extra)


# --------------------------------------------------------------------------- #
# MET: resolution near the injected ~20 GeV (no card formula -> sanity ceiling)
# --------------------------------------------------------------------------- #
def test_met_gate_passes(ctx):
    results = met.run(ctx)
    assert not _gate_passes(results)


def test_met_recovers_injected_resolution(ctx):
    results = {r.name: r for r in met.run(ctx)}
    res = results["level0.met.resolution"].measured
    # injected per-axis met_resolution_gev=20; recover within ~1 GeV statistics
    assert res == pytest.approx(20.0, abs=2.0)
