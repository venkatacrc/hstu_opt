#!/usr/bin/env bash
# Repair a broken fbgemm_gpu_hstu install (Python package present, .so missing).
# Usage: ./scripts/repair_hstu_so.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

print_env
echo "==> Forcing fbgemm_gpu_hstu rebuild (MAX_JOBS=${MAX_JOBS})"

# Drop skip markers by removing the broken package; 03 will rebuild step 4.
"${SCRIPT_DIR}/docker_run.sh" --privileged --name "${CONTAINER_NAME}-repair-hstu" -- \
  bash -lc "
    set -euo pipefail
    DEPS=${CONTAINER_RAID}/deps
    export PYTHONUSERBASE=\${DEPS}
    export PATH=\${DEPS}/bin:\$PATH
    pip uninstall -y fbgemm-gpu-hstu fbgemm_gpu_hstu hstu 2>/dev/null || true
    rm -rf \${DEPS}/lib/python3.12/site-packages/hstu \
           \${DEPS}/lib/python3.12/site-packages/hstu*dist-info \
           \${DEPS}/lib/python3.12/site-packages/fbgemm_gpu_hstu* \
           2>/dev/null || true
    echo 'purged broken hstu package'
  "

MAX_JOBS="${MAX_JOBS}" ./scripts/03_install_train.sh
