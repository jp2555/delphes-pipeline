#!/usr/bin/env bash
# Runs ON a worker node and reports what that node can actually see. Deliberately
# self-contained: it must not depend on /ceph, since whether /ceph exists is the
# question. Always exits 0 so the report comes back even when everything fails.
echo "HOST            $(hostname -f)"
echo "USER            $(id -un)"
echo "CPUS            $(nproc 2>/dev/null || echo '?')"
echo "MEM_GB          $(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo '?')"
echo "SCRATCH         ${_CONDOR_SCRATCH_DIR:-unset}"

# 1. is the shared area visible, and is the software environment reachable?
for d in /ceph /ceph/jpan /ceph/jpan/delphes-pipeline \
         /ceph/jpan/delphes-pipeline/.pixi/envs/nsbi-env-gpu/bin; do
    if [ -d "$d" ]; then echo "DIR             $d YES"; else echo "DIR             $d NO"; fi
done
PY=/ceph/jpan/delphes-pipeline/.pixi/envs/nsbi-env-gpu/bin/python
if [ -x "$PY" ] && "$PY" -c 'import uproot, awkward' 2>/dev/null; then
    echo "PIXI_ENV        USABLE"
else
    echo "PIXI_ENV        NOT USABLE"
fi

# 2. was the grid proxy forwarded, and is it still valid?
echo "X509_USER_PROXY ${X509_USER_PROXY:-unset}"
if [ -n "${X509_USER_PROXY:-}" ] && [ -r "${X509_USER_PROXY}" ]; then
    echo "PROXY_READABLE  YES"
else
    echo "PROXY_READABLE  NO"
fi
if command -v voms-proxy-info >/dev/null 2>&1; then
    echo "PROXY_TIMELEFT  $(voms-proxy-info -timeleft 2>/dev/null || echo 'n/a')"
    echo "VOMS_AC_LEFT    $(voms-proxy-info -actimeleft 2>/dev/null || echo 'n/a')"
else
    echo "PROXY_TOOLS     voms-proxy-info NOT on PATH"
fi

# 3. can this node actually read the data? the whole plan rests on this
if command -v xrdfs >/dev/null 2>&1; then
    echo "XRDFS           $(command -v xrdfs)"
    if timeout 90 xrdfs cmsdcache-kit-disk.gridka.de:1094 \
         stat /store/user/sdaigler/mc_production/delphes >/dev/null 2>&1; then
        echo "DCACHE_READ     YES"
    else
        echo "DCACHE_READ     NO"
    fi
else
    echo "XRDFS           NOT on PATH"
    echo "DCACHE_READ     UNTESTED"
fi
echo "CVMFS_CMS       $([ -d /cvmfs/cms.cern.ch ] && echo YES || echo NO)"

# 4. can it WRITE where the outputs are meant to go?
if touch /ceph/jpan/ntuples/.probe_write_$$ 2>/dev/null; then
    rm -f /ceph/jpan/ntuples/.probe_write_$$; echo "CEPH_WRITE      YES"
else
    echo "CEPH_WRITE      NO"
fi
exit 0
