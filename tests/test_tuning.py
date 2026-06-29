"""Tuning lens: residual + diagnostic + report, sharing the gate's measurements."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import build_ctx

from delphes_pipeline.tuning import report as treport
from delphes_pipeline.tuning import targets as T


def test_tuning_report_runs_and_writes(good_fixture_path):
    ctx = build_ctx(good_fixture_path)
    results = treport.run_tuning(ctx)
    by = {r.observable: r for r in results}

    # every diagnostic-map observable is covered
    assert set(by) == set(T.tuning_observables())

    # energy responses are unity on the fixture -> on target
    assert by["tau_energy_response"].status == "on_target"
    assert by["bjet_energy_response"].status == "on_target"

    # the m_bb peak (fixture ~125) misses the 116 GeV anchor -> flagged with its knob
    assert by["mbb_peak"].status == "needs_tuning"
    assert "b-jet" in by["mbb_peak"].knob.lower()

    # efficiencies have no digitised target dropped in yet
    assert by["btag_eff_b"].status == "no_target"

    out = Path(ctx.output_dir)
    assert (out / "tuning_report.md").exists()
    payload = json.loads((out / "tuning_report.json").read_text())
    assert len(payload["results"]) == len(results)


def test_every_result_has_a_card_knob_and_action(good_fixture_path):
    ctx = build_ctx(good_fixture_path)
    for r in treport.run_tuning(ctx):
        assert r.knob and r.action and r.note_section
