#!/usr/bin/env bash
# Force clean rebuild of fbgemm_gpu_hstu only (step 4 of 03_install_train.sh).
# Use when site-packages/hstu exists but fbgemm_gpu_experimental_hstu.so is missing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

require_cmd docker
export MAX_JOBS="${MAX_JOBS:-2}"

echo "==> Forcing clean hstu rebuild (MAX_JOBS=${MAX_JOBS})"
# Remove marker-free: delete installed package so 03 will rebuild
rm -rf \
  "${DEPS_ROOT}/lib/python3.12/site-packages/hstu" \
  "${DEPS_ROOT}/lib/python3.12/site-packages/hstu-"* \
  "${DEPS_ROOT}/lib/python3.12/site-packages/fbgemm_gpu_hstu"* \
  "${DEPS_ROOT}/local/lib/python3.12/dist-packages/hstu" \
  "${RECSYS_ROOT}/third_party/FBGEMM/fbgemm_gpu/experimental/hstu/build" \
  "${RECSYS_ROOT}/third_party/FBGEMM/fbgemm_gpu/experimental/hstu/dist" 2>/dev/null || true

"${SCRIPT_DIR}/03_install_train.sh"
