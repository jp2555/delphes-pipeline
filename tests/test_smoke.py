"""End-to-end gate: ``run_validation.run`` on the good and broken fixtures.

The example config enables only ``pilot_gate`` and ``level0`` (level1-4 are
stubs and stay disabled), so this exercises the full path config -> context ->
levels -> report on a card-injected good fixture (verdict PASS, exit 0) and on
the broken fixture (verdict FAIL, exit 1), and checks ``report.json`` is written.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import make_config

from delphes_pipeline.validation import run_validation


def _enabled_levels(config) -> set:
    return {name for name, blk in config["levels"].items() if blk.get("enabled")}


def test_smoke_good_passes(good_fixture_path):
    config = make_config(good_fixture_path)
    assert _enabled_levels(config) == {"pilot_gate", "level0", "ntuplizer"}

    report = run_validation.run(config)

    assert report.passed is True
    assert report.exit_code == 0
    assert not report.gate_failures

    report_json = Path(config["output"]["dir"]) / "report.json"
    assert report_json.exists()
    payload = json.loads(report_json.read_text())
    assert payload["passed"] is True
    assert payload["n_gate_failures"] == 0


def test_smoke_broken_fails(broken_fixture_path):
    config = make_config(broken_fixture_path)
    report = run_validation.run(config)

    assert report.passed is False
    assert report.exit_code == 1
    assert report.gate_failures

    report_json = Path(config["output"]["dir"]) / "report.json"
    assert report_json.exists()
    payload = json.loads(report_json.read_text())
    assert payload["passed"] is False
    assert payload["n_gate_failures"] >= 1
