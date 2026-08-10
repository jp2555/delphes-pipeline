"""The τ_h closure must bin by the variable Delphes applies the formula at.

Delphes ``TauTagging`` evaluates its ``EfficiencyFormula`` at the **reco jet**
kinematics. With the stock flat 0.6 formula the binning axis was immaterial, so a
gen-τ-pT closure passed; the pT-dependent v1 formula (PATCH-7) exposes the
mismatch — binning a jet-pT-applied formula by gen τ pT smears it through the jet
response and the low-pT matching turn-on, flattening the measured curve.

The fixture draws the τ tag at the jet pT (Delphes semantics) with a configurable
``tau_response`` = jet pT / gen τ pT, so the two axes are genuinely distinguishable
and the closure is only correct on one of them.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import build_ctx
from make_fixture import make_fixture

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.result import Severity
from delphes_pipeline.validation.level0_objects import tau
from delphes_pipeline.validation.references import card_formulas as cf

# the v1 τ formula — steep and pT-dependent, i.e. sensitive to the binning axis
_TAU_V1 = lambda pt, eta: float(cf.expected_v1("tau_eff", np.atleast_1d(pt), np.atleast_1d(eta))[0])
_TAU_MISTAG_V1 = lambda pt, eta: float(cf.expected_v1("tau_mistag", np.atleast_1d(pt), np.atleast_1d(eta))[0])
_RESPONSE = 0.90   # τ-jets under-measure the gen τ (real Delphes: ~0.92-0.99)
# The default binning's 200-300 GeV bin holds ~35 τ on a fixture this size (real
# data: ~7.5k), where a 3σ statistical swing reads as a closure failure on either
# axis. Stop at 150 GeV so the comparison measures the axis, not the sparse tail.
_BINS = [20, 30, 40, 50, 70, 100, 150]


@pytest.fixture(scope="module")
def tau_response_fixture(tmp_path_factory):
    """v1 τ formula applied at jet pT, with jet pT = 0.90 × gen τ pT."""
    path = tmp_path_factory.mktemp("tauaxis") / "signal_taus.root"
    make_fixture(
        str(path), n_events=30000, seed=5, tau_response=_RESPONSE,
        tau_eff=_TAU_V1, tau_mistag=_TAU_MISTAG_V1,
    )
    return str(path)


def _closure(ctx, x, bins=None):
    """Failing-bin fraction of the τ_h efficiency closure on axis ``x``."""
    from delphes_pipeline.core.closure import closure_from_profile
    prof = obs.tau_efficiency(ctx.events, bins=bins or _BINS, x=x)
    return closure_from_profile(ctx, prof, name=f"tmp.tau_eff.{x}")


def test_jet_pt_axis_closes_gen_pt_axis_does_not(tau_response_fixture):
    """The same events close on the jet-pT axis and fail on the gen-τ axis."""
    ctx = build_ctx(tau_response_fixture, card="cards/cms_card_v1.tcl")
    jet = _closure(ctx, "jet_pt")
    gen = _closure(ctx, "gen_pt")
    assert jet.passed, f"jet-pT closure should recover the card: {jet.detail}"
    assert not gen.passed, f"gen-pT closure should NOT close a jet-pT formula: {gen.detail}"


def test_level0_tau_leaf_uses_the_jet_pt_axis(tau_response_fixture):
    """The validation lens gates on the jet-pT axis, so this sample passes."""
    ctx = build_ctx(tau_response_fixture, card="cards/cms_card_v1.tcl")
    results = {r.name: r for r in tau.run(ctx)}
    eff = results["level0.tau.tau_eff"]
    assert eff.severity is Severity.GATE
    assert eff.passed, eff.detail


def test_tuning_lens_uses_the_jet_pt_axis(tau_response_fixture):
    """The tuning lens is on the jet axis too: the NanoAOD anchor is binned in
    GenVisTau (VISIBLE τ) pT, whose Delphes counterpart is the τ-jet — the full
    gen τ carries the ν as well, so that axis would read as an efficiency deficit."""
    from delphes_pipeline.tuning import targets

    ctx = build_ctx(tau_response_fixture, card="cards/cms_card_v1.tcl")
    prof = targets.PROFILE_OBSERVABLES["tau_eff"](ctx.events, obs.DEFAULT_PT_BINS)
    assert prof.xlabel.startswith("matched jet pT")
    assert prof.centers.size and np.all(np.isfinite(prof.values))


def test_gen_axis_would_read_as_a_deficit_vs_the_anchor(tau_response_fixture):
    """Why the axis matters for tuning: against a flat anchor-like target, the gen-τ
    axis shows a systematic negative residual that the jet axis does not."""
    ctx = build_ctx(tau_response_fixture, card="cards/cms_card_v1.tcl")
    dev = {}
    for x in ("jet_pt", "gen_pt"):
        prof = obs.tau_efficiency(ctx.events, bins=_BINS, x=x)
        target = cf.expected_v1("tau_eff", prof.centers, np.zeros_like(prof.centers))
        dev[x] = float(np.average(np.abs(prof.values / target - 1.0), weights=prof.counts))
    assert dev["jet_pt"] < 0.02, dev          # closes
    assert dev["gen_pt"] > 2 * dev["jet_pt"]  # a pure axis mismatch, read as a deficit


def test_axis_choice_is_validated():
    with pytest.raises(ValueError, match="gen_pt.*jet_pt"):
        obs.tau_efficiency(None, x="bogus")
