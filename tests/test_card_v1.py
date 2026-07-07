"""Card v1 (PATCH-6/7 Run-3 taggers): formulas, dispatch, and closure.

The v1 tagger formulas are smooth fits to the 2024 NanoAODv15 kl=1 anchor
measurement (derive_maps 2026-07-06, pasted verbatim below as ground truth).
These tests pin: (1) the fits still match that anchor, (2) the card-path
dispatch picks the right formula set, (3) the tcl and the python transcription
stay in lockstep, and (4) a sample produced *with* the v1 formulas closes
against the v1 target end-to-end — and fails against the v0 target, so the
dispatch is load-bearing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import build_ctx
from make_fixture import make_fixture

from delphes_pipeline.validation.level0_objects import btag, tau
from delphes_pipeline.validation.references import card_formulas as cf

_CARD_V1 = Path(__file__).resolve().parents[1] / "cards" / "cms_card_v1.tcl"

# derive_maps output on the 2024 NanoAODv15 kl=1 anchor (200k events),
# DEFAULT_PT_BINS [20,30,40,50,70,100,150,200,300] -> approx bin centers
_CENTERS = np.array([25.0, 35.0, 45.0, 60.0, 85.0, 125.0, 175.0, 250.0])
_ANCHOR = {
    "btag_eff_b": [0.763, 0.807, 0.829, 0.855, 0.868, 0.881, 0.892, 0.890],
    "btag_eff_c": [0.183, 0.153, 0.157, 0.117, 0.102, 0.100, 0.110, 0.103],
    "btag_mistag_light": [0.159, 0.132, 0.094, 0.058, 0.033, 0.023, 0.017, 0.019],
    "tau_eff": [0.388, 0.519, 0.579, 0.608, 0.637, 0.674, 0.714, 0.737],
}


@pytest.mark.parametrize("quantity,tol", [
    ("btag_eff_b", 0.015), ("btag_eff_c", 0.02),
    ("btag_mistag_light", 0.015), ("tau_eff", 0.03),
])
def test_v1_formulas_match_anchor(quantity, tol):
    eta = np.zeros_like(_CENTERS)
    got = cf.expected_v1(quantity, _CENTERS, eta)
    assert np.max(np.abs(got - np.asarray(_ANCHOR[quantity]))) < tol


def test_v1_tau_mistag_in_anchor_band():
    v = cf.expected_v1("tau_mistag", np.array([50.0]), np.array([0.0]))[0]
    assert 0.002 <= v <= 0.007


def test_for_card_dispatch():
    assert cf.for_card("cards/cms_card_v0.tcl") is cf.expected
    assert cf.for_card("/abs/path/cards/cms_card_v1.tcl") is cf.expected_v1
    assert cf.for_card("cms_card_V1.tcl") is cf.expected_v1          # case-insensitive
    assert cf.for_card("delphes_card_CMS_hhbbtt_v0.tcl") is cf.expected  # production name
    assert cf.for_card("unversioned_card.tcl") is cf.expected        # fixtures: stock


@pytest.mark.parametrize("name", ["cms_card_v2.tcl", "cms_card_v10.tcl", "card_nanoaodv15.tcl"])
def test_for_card_rejects_untranscribed_versions(name):
    """A versioned card without transcribed formulas must fail loudly, not
    silently validate against another card's targets."""
    with pytest.raises(ValueError, match="no transcribed closure formulas"):
        cf.for_card(name)


def test_v1_lepton_blocks_defer_to_v0():
    pt, eta = np.array([30.0]), np.array([0.5])
    for q in ("electron_eff", "muon_eff"):
        assert cf.expected_v1(q, pt, eta) == cf.expected(q, pt, eta)


def test_v1_tcl_lockstep():
    """The tcl carries the exact formulas expected_v1 transcribes."""
    tcl = _CARD_V1.read_text()
    for snippet in (
        "{0.019 + 0.32*exp(-pt/30.0)}",          # PATCH-6 light
        "{0.094 + 2.22/pt}",                     # PATCH-6 c
        "{(pt > 4.0) * (0.904 - 3.53/pt)}",      # PATCH-6 b
        "{0.004}",                               # PATCH-7 mistag
        "{(pt > 12.5) * (0.776 - 9.7/pt)}",      # PATCH-7 tau
        "set DeltaR 0.4",                        # PATCH-7 match radius
    ):
        assert snippet in tcl, f"card/formula drift: {snippet!r} not in {_CARD_V1.name}"


def _v1fn(quantity):
    return lambda pt, eta: float(cf.expected_v1(quantity, np.atleast_1d(pt), np.atleast_1d(eta))[0])


@pytest.fixture(scope="module")
def v1_fixture_path(tmp_path_factory):
    """Fixture whose objects are tagged at the v1 card formulas."""
    path = tmp_path_factory.mktemp("v1") / "signal_v1.root"
    make_fixture(
        str(path), n_events=8000, seed=11, met_resolution_gev=20, mbb_width_gev=10,
        btag_eff=_v1fn("btag_eff_b"), ctag_eff=_v1fn("btag_eff_c"),
        ltag_mistag=_v1fn("btag_mistag_light"),
        tau_eff=_v1fn("tau_eff"), tau_mistag=_v1fn("tau_mistag"),
        electron_eff=_v1fn("electron_eff"), muon_eff=_v1fn("muon_eff"),
    )
    return str(path)


def _gate_failures(results):
    from delphes_pipeline.core.result import Severity
    return [r.name for r in results if r.severity is Severity.GATE and not r.passed]


def test_v1_sample_closes_against_v1(v1_fixture_path):
    """ALL tagger closures pass — including btag_eff_c and tau_mistag, which the
    v1 severity map no longer hides behind the gate filter."""
    ctx = build_ctx(v1_fixture_path, card="cards/cms_card_v1.tcl")
    results = btag.run(ctx) + tau.run(ctx)
    assert not [r.name for r in results if not r.passed]


def test_v1_sample_fails_against_v0(v1_fixture_path):
    """The dispatch is load-bearing: v1-tagged objects miss the v0 target."""
    ctx = build_ctx(v1_fixture_path, card="cards/cms_card_v0.tcl")
    assert _gate_failures(btag.run(ctx)) + _gate_failures(tau.run(ctx))


def test_v1_severity_map_is_card_aware(v1_fixture_path):
    """v1 gates btag_eff_c (stock-formula waiver obsolete); tau_mistag stays
    warn (per-parton mechanism unchanged by PATCH-7); v0 keeps its waivers."""
    from delphes_pipeline.core.result import Severity

    v1 = {r.name: r for r in btag.run(build_ctx(v1_fixture_path, card="cards/cms_card_v1.tcl"))}
    assert v1["level0.btag.btag_eff_c"].severity is Severity.GATE
    v1_tau = {r.name: r for r in tau.run(build_ctx(v1_fixture_path, card="cards/cms_card_v1.tcl"))}
    assert v1_tau["level0.tau.tau_mistag"].severity is Severity.WARN

    v0 = {r.name: r for r in btag.run(build_ctx(v1_fixture_path, card="cards/cms_card_v0.tcl"))}
    assert v0["level0.btag.btag_eff_c"].severity is Severity.WARN
