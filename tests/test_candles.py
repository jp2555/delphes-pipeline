"""Level-1 candles: tt̄ in-situ b-tag closure + Z→ττ visible-mass scaffold."""

from __future__ import annotations

import numpy as np
from conftest import build_ctx
from make_candle_fixture import make_dy_fixture, make_dy_fixture_nu, make_ttbar_fixture

from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.validation.level1_candles import selections, ttbar, ztautau
from delphes_pipeline.validation.level1_candles import run as level1_run
from delphes_pipeline.validation.references import card_formulas


def test_eb_insitu_recovers_injected(tmp_path):
    p = tmp_path / "tt.root"
    make_ttbar_fixture(str(p), n_events=6000, seed=1, btag_eff=0.70)
    ev = DelphesEvents(str(p))
    eb, N1, N2, bb_pt = selections.eb_insitu(ev, selections.emu_os_mask(ev))
    assert abs(eb - 0.70) < 0.03           # 2N2/(N1+2N2) recovers the per-jet eff
    assert N1 > 0 and N2 > 0 and bb_pt.size > 0


def test_ttbar_candle_gate_passes_when_eb_matches_card(good_fixture_path, tmp_path):
    # inject the card b-eff at a representative pT so the in-situ closure matches
    eff = float(card_formulas.expected("btag_eff_b", np.array([80.0]), np.array([0.0]))[0])
    p = tmp_path / "tt.root"
    make_ttbar_fixture(str(p), n_events=8000, seed=2, btag_eff=eff)
    ctx = build_ctx(good_fixture_path)
    results = {r.name: r for r in ttbar.run(ctx, DelphesEvents(str(p)))}

    r = results["level1.ttbar.eb_insitu_closure"]
    assert r.severity.value == "gate"
    assert np.isfinite(r.measured)
    assert r.passed, f"in-situ ε_b {r.measured} should match the card input {r.target}"


def test_ztautau_candle_fastmtt_peaks_at_mz(good_fixture_path, tmp_path):
    p = tmp_path / "dy.root"
    make_dy_fixture_nu(str(p), n_events=6000, seed=3)
    ctx = build_ctx(good_fixture_path)            # production default met_sigma=25, tol=10
    by = {r.name: r for r in ztautau.run(ctx, DelphesEvents(str(p)))}

    # the visible peak sits below m_Z (neutrinos missing)
    assert by["level1.ztautau.visible_peak"].measured < 75.0
    # the FastMTT estimator restores the m_Z peak (robust mode) and the GATE passes
    r = by["level1.ztautau.peak_at_mZ"]
    assert r.severity.value == "gate"
    assert abs(r.measured - 91.2) < 8.0
    assert r.passed
    assert "tail" in r.detail and r.extra["n_pairs"] > 100


def test_level1_aggregates_both_candles(good_fixture_path, tmp_path):
    tt = tmp_path / "tt.root"; dy = tmp_path / "dy.root"
    make_ttbar_fixture(str(tt), n_events=3000, seed=4)
    make_dy_fixture(str(dy), n_events=3000, seed=5)
    ctx = build_ctx(good_fixture_path)
    ctx.config["candles"] = {"ttbar": str(tt), "ztautau": str(dy)}

    names = {r.name for r in level1_run(ctx)}
    assert "level1.ttbar.eb_insitu_closure" in names
    assert "level1.ztautau.visible_peak" in names


def test_level1_notes_unconfigured_candle(good_fixture_path):
    ctx = build_ctx(good_fixture_path)
    ctx.config["candles"] = {}  # nothing configured
    names = {r.name for r in level1_run(ctx)}
    assert "level1.ttbar.not_configured" in names
    assert "level1.ztautau.not_configured" in names
