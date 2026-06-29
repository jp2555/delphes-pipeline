#!/bin/bash
# Run the object-tuning lens: measure object response on a sample, compare to the
# POG/anchor targets, and write tuning_report.{md,json} + measured-vs-target
# overlays under <output.dir>. Re-run after each card edit to drive the loop.
#
#   bash runners/run_tuning.sh [config.yml] [--delphes-root DIR] [--max-events N]
set -uo pipefail

PYTHON="${PYTHON:-pixi run python}"          # Perlmutter bare python is 2.7
CONFIG="${1:-delphes_pipeline/validation/config.example.yml}"

$PYTHON -m delphes_pipeline.tuning.run_tuning --config "$CONFIG" "${@:2}"
