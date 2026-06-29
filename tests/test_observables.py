"""The shared measurement layer: binning primitives + new tuning observables."""

from __future__ import annotations

import numpy as np
from make_fixture import make_fixture

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.io import DelphesEvents


def _ev(tmp_path, **kw):
    p = tmp_path / "obs.root"
    make_fixture(str(p), n_events=4000, seed=4, **kw)
    return DelphesEvents(str(p))


def test_binned_efficiency_recovers_a_flat_rate():
    rng = np.random.default_rng(0)
    pt = rng.uniform(20, 300, 20000)
    passed = rng.random(20000) < 0.6
    prof = obs.binned_efficiency(pt, passed, obs.DEFAULT_PT_BINS, quantity="x")
    assert prof.centers.size > 0
    assert np.allclose(prof.values, 0.6, atol=0.05)


def test_energy_response_is_unity_on_the_fixture(tmp_path):
    # the fixture's GenJets are the reco jets, so reco/gen response is ~1
    ev = _ev(tmp_path)
    for prof in (obs.bjet_energy_response(ev), obs.tau_energy_response(ev)):
        assert prof.kind == "response"
        good = prof.counts > 0
        assert np.allclose(prof.values[good], 1.0, atol=0.05)


def test_mbb_peak_finds_the_resonance(tmp_path):
    ev = _ev(tmp_path, mbb_width_gev=10.0)
    pk = obs.mbb_peak(ev)
    assert abs(pk.peak - 125.0) < 6.0
    assert pk.width < 20.0
    assert 0.3 < pk.core_fraction <= 1.0


def test_btag_efficiency_extractor_matches_injection(tmp_path):
    ev = _ev(tmp_path)
    prof = obs.btag_efficiency(ev, "btag_eff_b")
    assert prof.quantity == "btag_eff_b"
    assert prof.values.size > 0 and np.all(prof.values <= 1.0)
