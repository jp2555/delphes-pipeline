"""Entry point: read a config, run the enabled validation levels, write the report.

    python -m delphes_pipeline.validation.run_validation --config <config.yml>

The process exit code is the gate: ``0`` iff every GATE-severity check passed.
Each level is a module exposing ``run(ctx: ValidationContext) -> list[CheckResult]``;
levels are dispatched by name from the config's ``levels:`` block.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import yaml

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.core.references import ReferenceStore
from delphes_pipeline.core.report import Report
from delphes_pipeline.core import provenance as prov

# level name -> "module:attr" providing run(ctx) -> list[CheckResult]
LEVEL_RUNNERS: dict[str, str] = {
    "pilot_gate": "delphes_pipeline.validation.pilot_gate.checks:run",
    "level0": "delphes_pipeline.validation.level0_objects:run",
    "ntuplizer": "delphes_pipeline.validation.ntuplizer_check:run",
    "level1": "delphes_pipeline.validation.level1_candles:run",
    "level2": "delphes_pipeline.validation.level2_signal:run",
    "level3": "delphes_pipeline.validation.level3_analysis:run",
    "level4": "delphes_pipeline.validation.level4_kappa:run",
}


def _load_runner(spec: str):
    mod_name, attr = spec.split(":")
    return getattr(importlib.import_module(mod_name), attr)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_context(config: dict[str, Any], *, max_events: int | None) -> ValidationContext:
    cfg_input = config["input"]
    delphes_path = cfg_input["delphes_root"]
    treename = cfg_input.get("treename", "Delphes")
    entry_stop = max_events if max_events is not None else cfg_input.get("max_events")

    events = DelphesEvents(delphes_path, treename=treename, entry_stop=entry_stop)

    output_dir = Path(config.get("output", {}).get("dir", "outputs/validation"))
    plot_dir = output_dir / "plots"

    # inject the card-formula closure target into the reference store
    from delphes_pipeline.validation.references import card_formulas

    ref_dir = config.get("references", {}).get("dir", "delphes_pipeline/validation/references/data")
    references = ReferenceStore(ref_dir, card_formula_fn=card_formulas.expected)

    provenance = prov.collect(
        card_path=config.get("card", "cards/cms_card_v0.tcl"),
        input_path=delphes_path,
        n_events=events.n,
        config=config,
    )

    return ValidationContext(
        config=config,
        events=events,
        references=references,
        output_dir=output_dir,
        plot_dir=plot_dir,
        provenance=provenance,
    )


def run(config: dict[str, Any], *, max_events: int | None = None) -> Report:
    ctx = build_context(config, max_events=max_events)
    report = Report(provenance=ctx.provenance)

    levels_cfg = config.get("levels", {})
    for name, spec in LEVEL_RUNNERS.items():
        if not levels_cfg.get(name, {}).get("enabled", False):
            continue
        try:
            runner = _load_runner(spec)
        except Exception as exc:  # stub levels may not provide run yet
            print(f"[skip] level {name!r}: not runnable ({exc})", file=sys.stderr)
            continue
        print(f"[run ] level {name!r} …", file=sys.stderr)
        report.add(runner(ctx))

    json_path, md_path = report.write(ctx.output_dir)
    print(f"[done] {len(report.results)} checks → {json_path}")
    print(f"[done] verdict: {'PASS' if report.passed else 'FAIL'} "
          f"({len(report.gate_failures)} gate failures) → {md_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Delphes card-validation gate")
    ap.add_argument("--config", required=True, help="path to the YAML config")
    ap.add_argument("--max-events", type=int, default=None,
                    help="override: read only the first N events (fast run)")
    ap.add_argument("--output-dir", default=None, help="override output.dir")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    if args.output_dir:
        config.setdefault("output", {})["dir"] = args.output_dir

    report = run(config, max_events=args.max_events)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
