#!/bin/bash
# Produce the validation + signal-baseline figures (object spectra, m_bb, visible
# m_tautau, gen m_HH across the six kappa_lambda points, lepton-floor curves).
# Figures land in <output.dir>/figures. The m_HH overlay reads the six sample
# directories under DIHIGGS_RAW (default /pscratch/sd/j/jing/dihiggs/raw).
#
#   bash runners/make_validation_plots.sh [config.yml] [extra args...]
set -uo pipefail

PYTHON="${PYTHON:-pixi run python}"          # Perlmutter bare python is 2.7
CONFIG="${1:-delphes_pipeline/validation/config.example.yml}"
KLBASE="${DIHIGGS_RAW:-/pscratch/sd/j/jing/dihiggs/raw}"

$PYTHON -m delphes_pipeline.plots.make_plots --config "$CONFIG" --klambda-base "$KLBASE" "${@:2}"
