#!/usr/bin/env bash
# Answer the "what can the workers see?" questions empirically instead of by email.
#   PROXY="$HOME/.globus/x509up" bash runners/probe.sh [n_jobs]
# Then:  cat /ceph/jpan/ntuples/_probe/*.out
set -euo pipefail
N="${1:-12}"
OUT="${OUT:-/ceph/jpan/ntuples/_probe}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROXY="${PROXY:-}"
mkdir -p "$OUT"

# NOTE: no `should_transfer_files = NO` here. That is what pinned the real jobs to the
# submit node: it makes Condor require TARGET.FileSystemDomain == MY.FileSystemDomain,
# and every machine in this pool advertises its OWN hostname as its domain.
cat > "$OUT/probe.sub" <<EOF
universe                = vanilla
executable              = ${REPO}/runners/probe_worker.sh
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_executable     = true
$([ -n "$PROXY" ] && printf 'x509userproxy          = %s\nuse_x509userproxy       = true' "$PROXY")
request_cpus            = 1
request_memory          = 2 GB
request_disk            = 1 GB
output                  = ${OUT}/probe.\$(Process).out
error                   = ${OUT}/probe.\$(Process).err
log                     = ${OUT}/probe.log
queue ${N}
EOF
echo "submitting ${N} probes -> ${OUT}"
condor_submit "$OUT/probe.sub"
cat <<EOF

When they finish:
  cat ${OUT}/probe.*.out | sort | uniq -c | sort -rn   # what most workers report
  grep -h '^HOST\|^DIR .*delphes-pipeline \|^DCACHE_READ\|^PIXI_ENV' ${OUT}/probe.*.out
EOF
