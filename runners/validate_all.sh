#!/bin/bash
# Validate the Delphes card across all six kappa_lambda signal points.
# Each point is a separate Delphes sample directory; the card is identical, but
# the kinematics shift with kappa_lambda (toward threshold), so the object
# response is worth checking per point. Writes one report per point under
# outputs/validation/kl-<value>/ and a non-zero exit if any point fails its gate.
#
#   bash runners/validate_all.sh
set -uo pipefail

BASE="${DIHIGGS_RAW:-/pscratch/sd/j/jing/dihiggs/raw}"
PREFIX="GluGluHHto2B2Tau_Par-c2-0p00-kl"
SUFFIX="-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8_Delphes"
CONFIG="${1:-delphes_pipeline/validation/config.example.yml}"

# Use the project's pixi env (Perlmutter's bare `python` is 2.7). Override with
# e.g. PYTHON=python3 if you manage the environment yourself.
PYTHON="${PYTHON:-pixi run python}"

rc=0
for KL in m1p00 0p00 1p00 2p45 3p00 5p00; do
  DIR="${BASE}/${PREFIX}-${KL}${SUFFIX}"
  echo "================ kl=${KL} ================"
  $PYTHON -m delphes_pipeline.validation.run_validation \
    --config "$CONFIG" \
    --delphes-root "$DIR" \
    --output-dir "outputs/validation/kl-${KL}"
  status=$?
  if [ $status -ne 0 ]; then
    echo "  >> kl=${KL} FAILED the gate (exit $status)"
    rc=1
  fi
done

echo "================ summary ================"
echo "per-point reports under outputs/validation/kl-*/ ; overall exit ${rc}"
exit $rc
