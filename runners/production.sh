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

# Production subtrees are PER SAMPLE and the hash alone is not a portable key:
# 6d2d1cb0 is the signal *test* subtree AND the ttbar/DY v0 one, so always pair the hash
# with _Delphes_v1/. Spanning two subtrees double-counts (they hold the same events).
#   full production:  signal 61fd1c12 (20k files, ~392 GB, 5M evt per kappa_lambda)
#                     ttbar  2ff38f65 (~4k files, 7.6-8.8 TB, 50M evt per channel)
SUB_SIGNAL="${SUB_SIGNAL:-_Delphes_v1/delphes-tree-61fd1c12}"
SUB_TTBAR="${SUB_TTBAR:-_Delphes_v1/delphes-tree-2ff38f65}"

# Grid proxy, needed only when SRC is root:// . voms-proxy-init writes
# /tmp/x509up_u$(id -u), which the worker nodes cannot see, so copy it somewhere shared:
#   voms-proxy-init --voms cms --valid 192:00
#   install -m 600 /tmp/x509up_u$(id -u) "$HOME/.x509up_production"
# NB the VOMS *attribute* lifetime is often capped well below the proxy lifetime; if the
# attribute expires mid-campaign, dCache auth fails even though the proxy looks valid.
PROXY="${PROXY:-}"
PROXYARG=""; [ -n "${PROXY}" ] && PROXYARG="--proxy ${PROXY}"

# MEASURED on a real ttbar shard: peak RSS is ~1.3x the shard's INPUT size (25.9 GB for a
# 22 GB shard, i.e. ~87 kB/event resident), because the reader concatenates the shard and
# holds the gen record. So the slot has to be sized off SHARD_GB, not guessed. The job is
# CPU-bound (102% CPU on a remote read), so fewer, larger shards cost nothing in I/O.
SHARD_GB="${SHARD_GB:-20}"
MEMORY="${MEMORY:-$(python3 -c "print(f'{max(4, round(${SHARD_GB}*1.3+6))} GB')")}"

echo "=== 1/3  maps: signal ==="
$RUN -m delphes_pipeline.tuning.derive_maps --config config.v1.yml \
     --output cards/tuning/maps_v1.json --max-events "${CAP}"

echo "=== 2/3  maps: ttbar ==="
$RUN -m delphes_pipeline.tuning.derive_maps --config config.ttbar.yml \
     --output cards/tuning/maps_ttbar_v1.json --max-events "${CAP}"

echo "=== 3/3  plan the shards ==="
$RUN scripts/make_shards.py \
    --sample signal "${SRC}/*kl-*" cards/tuning/maps_v1.json      "${SUB_SIGNAL}" \
    --sample ttbar  "${SRC}/*TT*"   cards/tuning/maps_ttbar_v1.json "${SUB_TTBAR}" \
    --out "${OUT}" --shard-gb "${SHARD_GB}" --memory "${MEMORY}" ${PROXYARG}

cat <<EOF

Planned with SHARD_GB=${SHARD_GB}, request_memory=${MEMORY}

Next:
  condor_submit ${OUT}/_plan/ntuplize.sub
  ${RUN} scripts/make_shards.py --verify ${OUT}     # after the jobs finish
EOF
