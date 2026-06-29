#!/bin/bash
# Validate the Delphes card across the kappa_lambda signal points (0, 1, 5).
# Each point is a separate Delphes sample directory; the card is identical, but
# the kinematics shift with kappa_lambda, so the object response is worth checking
# per point. Writes one report per point under outputs/validation/kl-<value>/ and
# a non-zero exit if any point fails its gate.
#
# Sample dirs are matched by the glob "<DELPHES_BASE>/*kl-<tag>*" so the exact
# directory name does not matter. Override the base / points with env vars:
#   DELPHES_BASE=/ceph/jpan/cms_nanoaod_2024_hh2b2tau/delphes
#   KL_POINTS="0p00 1p00 5p00"
#
#   bash runners/validate_all.sh
set -uo pipefail

DELPHES_BASE="${DELPHES_BASE:-/ceph/jpan/cms_nanoaod_2024_hh2b2tau/delphes}"
KL_POINTS="${KL_POINTS:-0p00 1p00 5p00}"
CONFIG="${1:-delphes_pipeline/validation/config.example.yml}"
PYTHON="${PYTHON:-pixi run python}"      # bare `python` may be 2.7 on some clusters

rc=0
for KL in $KL_POINTS; do
  echo "================ kl=${KL} ================"
  $PYTHON -m delphes_pipeline.validation.run_validation \
    --config "$CONFIG" \
    --delphes-root "${DELPHES_BASE}/*kl-${KL}*" \
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
