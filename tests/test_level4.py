"""Level 4: κ_λ-critical m_HH faithfulness on synthetic HH→bb̄ττ signal."""

from __future__ import annotations

from conftest import build_ctx
from make_signal_fixture import make_hhbbtt_fixture

from delphes_pipeline.validation import level4_kappa as L4


def _by(path):
    return {r.name: r for r in L4.run(build_ctx(path))}


def test_level4_passes_on_faithful_signal(tmp_path):
    p = tmp_path / "hh.root"
    make_hhbbtt_fixture(str(p), n_events=5000, seed=1, bjet_smear=0.10)
    r = _by(str(p))
    assert r["level4.mhh.scale"].passed and abs(r["level4.mhh.scale"].measured - 1.0) < 0.10
    assert r["level4.mhh.resolution"].passed and r["level4.mhh.resolution"].measured < 0.20
    assert r["level4.mhh.acceptance_efficiency"].measured > 0.8     # fixture has all objects
    assert r["level4.mhh.migration_diagonal"].measured > 0.5        # mostly diagonal


def test_level4_scale_gate_fails_on_biased_bjets(tmp_path):
    p = tmp_path / "hh.root"
    make_hhbbtt_fixture(str(p), n_events=4000, seed=2, bjet_response=0.75)  # b-jets 25% low
    r = _by(str(p))
    assert r["level4.mhh.scale"].severity.value == "gate"
    assert not r["level4.mhh.scale"].passed       # m_HH biased low -> scale gate fails


def test_level4_resolution_gate_fails_on_smeared(tmp_path):
    p = tmp_path / "hh.root"
    make_hhbbtt_fixture(str(p), n_events=4000, seed=3, bjet_smear=1.0)      # washed-out m_HH
    r = _by(str(p))
    assert r["level4.mhh.resolution"].severity.value == "gate"
    assert not r["level4.mhh.resolution"].passed  # resolution gate fails


def test_level4_gen_mhh_robust_to_pythia_copies(tmp_path):
    """gen m_HH uses the hard Higgs, so b/τ shower copies + low-status Higgs copies in the
    full Pythia record don't corrupt the denominator (a leading-pT b/τ selection would)."""
    p = tmp_path / "hh.root"
    make_hhbbtt_fixture(str(p), n_events=5000, seed=1, bjet_smear=0.10, add_copies=True)
    r = _by(str(p))
    assert r["level4.mhh.scale"].passed and abs(r["level4.mhh.scale"].measured - 1.0) < 0.10
    assert r["level4.mhh.resolution"].passed


def test_level4_reports_no_gen_on_non_signal(good_fixture_path):
    # the good fixture has no gen Higgs -> Level 4 cannot reconstruct gen m_HH
    assert "level4.mhh.no_gen" in _by(good_fixture_path)
