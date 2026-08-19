#!/usr/bin/env bash
# UNTUNED baseline: plan the sharded ntuplization with NO tuning maps at all.
#
#   bash runners/production_untuned.sh /ceph/jpan/ntuples_untuned
#
# Why a separate runner and a separate output directory: stock Delphes is a DIFFERENT
# forward model from a tuned run. Provenance uniformity -- one frozen forward model behind
# every density in a fit -- is what the downstream unbinned CI depends on, so the two must
# never share a directory where a later merge could mix them. `maps none` is spelled out
# in the plan (maps_sha = "untuned") rather than implied by omission.
#
# This baseline sidesteps the open tuning questions entirely (per-process map application,
# missing parameterisation variables -- see docs/tuning_for_nsbi_audit.md) at the cost of
# a detector that is not CMS-like. That is a valid trade for a Delphes-INTERNAL NSBI
# comparison; it is not a basis for anything CMS-anchored.
set -euo pipefail

OUT="${1:-/ceph/jpan/ntuples_untuned}"
ENVN="${ENVN:-nsbi-env-gpu}"
RUN="pixi run -e ${ENVN} python"

SRC="${SRC:-root://cmsdcache-kit-disk.gridka.de:1094//store/user/sdaigler/mc_production/delphes}"

# One subtree per sample; the hash alone is not a portable key, so pair it with _Delphes_v1/.
# Spanning two subtrees double-counts (they hold the same events) -- make_shards refuses.
SUB_SIGNAL="${SUB_SIGNAL:-_Delphes_v1/delphes-tree-61fd1c12}"
SUB_BKG="${SUB_BKG:-_Delphes_v1/delphes-tree-2ff38f65}"

# SRC above is a dCache URL, so a grid proxy the WORKERS can read is MANDATORY. It must
# NOT live in /tmp: that is per-node, so the submit host's copy is invisible to mdm*/ms*.
#   voms-proxy-init --voms cms --valid 192:00
#   install -m 600 /tmp/x509up_u$(id -u) /ceph/$USER/.x509up
#   PROXY=/ceph/$USER/.x509up bash runners/production_untuned.sh /ceph/jpan/ntuples_untuned
# Set SRC to a local path instead if you have one; then no proxy is needed.
PROXY="${PROXY:-}"
PROXYARG=""
if [ -n "${PROXY}" ]; then
    PROXYARG="--proxy ${PROXY}"
    [ -r "${PROXY}" ] || { echo "PROXY=${PROXY} is not readable" >&2; exit 2; }
fi

SHARD_GB="${SHARD_GB:-20}"
# Files per shard. The byte cap alone assumes constant bytes-per-event, which low-mass
# DY breaks badly: its events are soft and compress well, so a 20 GB shard held enough
# of them to exhaust even a 45 GB slot. 60 bounds the concatenate regardless.
SHARD_FILES="${SHARD_FILES:-60}"
# Measured: peak RSS ~1.3x the shard's INPUT size (22.0 GB mean / 26.7 GB max at 20 GB
# shards). Request 30 GB, never below the measured mean -- dropping it to 16 GB held 20
# running jobs mid-flight.
MEMORY="${MEMORY:-30 GB}"

# DY is jet-binned (0J/1J/2J) plus a low-mass bin, and EACH CARRIES ITS OWN CROSS SECTION.
# They must be planned as separate samples: globbing them together loses the bin identity
# and with it the weight, exactly as globbing the kl directories lost kappa_lambda.
$RUN scripts/make_shards.py \
    --sample signal  "${SRC}/*kl-*"                    none "${SUB_SIGNAL}" \
    --sample ttbar   "${SRC}/*TT*"                     none "${SUB_BKG}" \
    --sample dy_0j   "${SRC}/*DYto2Tau*Bin-0J-MLL-50*" none "${SUB_BKG}" \
    --sample dy_1j   "${SRC}/*DYto2Tau*Bin-1J-MLL-50*" none "${SUB_BKG}" \
    --sample dy_2j   "${SRC}/*DYto2Tau*Bin-2J-MLL-50*" none "${SUB_BKG}" \
    --sample dy_low  "${SRC}/*DYto2Tau*Bin-MLL-10to50*" none "${SUB_BKG}" \
    --out "${OUT}" --shard-gb "${SHARD_GB}" --shard-files "${SHARD_FILES}" \
    --memory "${MEMORY}" ${PROXYARG} \
    ${REFRESH:+--refresh}

cat <<EOF

Planned UNTUNED with SHARD_GB=${SHARD_GB}, request_memory=${MEMORY}

Next:
  condor_submit ${OUT}/_plan/ntuplize.sub
  ${RUN} scripts/make_shards.py --verify ${OUT}                    # after the jobs finish
  ${RUN} scripts/merge_shards.py --out ${OUT} --target-gb 5 --jobs 0

Then convert to the SBI input format (one file per process):
  ${RUN} scripts/delphes_to_sbi.py --ntuple ${OUT}/merged --sample signal \\
      --out saved_datasets/dihiggs_delphes_data.root
  ${RUN} scripts/delphes_to_sbi.py --ntuple '${OUT}/merged/ttbar.*.parquet' \\
      --sample ttbar --out saved_datasets/ttbar_delphes_data.root
  ${RUN} scripts/delphes_to_sbi.py --ntuple '${OUT}/merged/dy_*.parquet' \\
      --sample dy --out saved_datasets/dy_delphes_data.root
EOF
