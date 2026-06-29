"""Entry point: measure object response on a sample and emit the tuning report.

    python -m delphes_pipeline.tuning.run_tuning --config <config.yml> [--delphes-root DIR]

Reuses the validation context builder (same reader, references, provenance), then
runs the tuning lens. Writes tuning_report.{md,json} + measured-vs-target overlays
under output.dir. Unlike the gate, this never sets a non-zero exit code — tuning
is iterative, not pass/fail.
"""

from __future__ import annotations

import argparse

from delphes_pipeline.validation.run_validation import build_context, load_config
from . import report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Delphes object-tuning report")
    ap.add_argument("--config", required=True, help="path to the YAML config")
    ap.add_argument("--delphes-root", default=None, help="override input.delphes_root")
    ap.add_argument("--max-events", type=int, default=None, help="read only the first N events")
    ap.add_argument("--output-dir", default=None, help="override output.dir")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    if args.delphes_root:
        config.setdefault("input", {})["delphes_root"] = args.delphes_root
    if args.output_dir:
        config.setdefault("output", {})["dir"] = args.output_dir

    ctx = build_context(config, max_events=args.max_events)
    report.run_tuning(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
