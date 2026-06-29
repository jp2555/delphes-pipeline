"""Plots lens: every figure renders on the fixture without error."""

from __future__ import annotations

from pathlib import Path

from make_fixture import make_fixture

from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.plots import figures as F


def _ev(tmp_path):
    p = tmp_path / "plots.root"
    make_fixture(str(p), n_events=2000, seed=6)
    return DelphesEvents(str(p))


def test_formula_figure_needs_no_data(tmp_path):
    out = Path(F.lepton_eff_floor_curves(tmp_path))
    assert out.exists() and out.stat().st_size > 0


def test_data_figures_render(tmp_path):
    ev = _ev(tmp_path)
    for fn in (F.jet_pt_spectrum, F.tauh_pt_spectrum, F.jet_eta_spectrum, F.mbb_peak_figure,
               F.lepton_pt_spectra, F.mtautau_figure, F.multiplicity_figure):
        out = Path(fn(ev, tmp_path))
        assert out.exists() and out.stat().st_size > 0


def test_mhh_overlay_handles_missing_gen_higgs(tmp_path):
    # the fixture has no gen Higgs; the overlay must render (empty) without crashing
    ev = _ev(tmp_path)
    out = Path(F.mhh_klambda_overlay({"kl=1": ev}, tmp_path))
    assert out.exists()
