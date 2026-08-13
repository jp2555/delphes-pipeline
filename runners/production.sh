#!/usr/bin/env bash
# Full-production driver: derive per-process maps, then plan the sharded ntuplization.
#
#   bash runners/production.sh /ceph/jpan/ntuples
#
# Two steps, deliberately separate: the maps are a CALIBRATION (bounded, one process each,
# minutes) while the ntuplization is the volume pass (memory-bound, sharded). Deriving
# per process is not optional — the tt̄ lepton scale factors differ from the signal's by
# up to 20%, and the fake-tau response is a jet-substructure property that does not
# transfer from a quark-rich signal to a gluon-rich background.
set -euo pipefail

OUT="${1:-/ceph/jpan/ntuples}"
ENVN="${ENVN:-nsbi-env-gpu}"
CAP="${CAP:-200000}"          # anchor/derivation cap; the maps do not need more
RUN="pixi run -e ${ENVN} python"

# Where the FULL production lives. Default is the local /ceph copy; set SRC to the dCache
# area to stream instead (GridKa is KIT's own storage, so this is a LAN read, and the
# 35-leaf whitelist means only a few percent of those 32 TB is ever transferred).
#   SRC='root://cmsdcache-kit-disk.gridka.de:1094//store/user/sdaigler/mc_production/delphes'
SRC="${SRC:-/ceph/jpan/gen-delphes}"
# MANDATORY when SRC holds more than one production subtree: they carry the SAME events,
# so spanning them double-counts. make_shards refuses rather than guessing.
SUBTREE="${SUBTREE:-}"
SUBARG=""; [ -n "${SUBTREE}" ] && SUBARG="--subtree ${SUBTREE}"

echo "=== 1/3  maps: signal ==="
$RUN -m delphes_pipeline.tuning.derive_maps --config config.v1.yml \
     --output cards/tuning/maps_v1.json --max-events "${CAP}"

echo "=== 2/3  maps: ttbar ==="
$RUN -m delphes_pipeline.tuning.derive_maps --config config.ttbar.yml \
     --output cards/tuning/maps_ttbar_v1.json --max-events "${CAP}"

echo "=== 3/3  plan the shards ==="
$RUN scripts/make_shards.py \
    --sample signal "${SRC}/*kl-*Delphes_v1" cards/tuning/maps_v1.json \
    --sample ttbar  "${SRC}/*TT*Delphes_v1"  cards/tuning/maps_ttbar_v1.json \
    --out "${OUT}" --shard-events "${SHARD_EVENTS:-150000}" ${SUBARG}

cat <<EOF

Next:
  condor_submit ${OUT}/_plan/ntuplize.sub
  ${RUN} scripts/make_shards.py --verify ${OUT}     # after the jobs finish
EOF
