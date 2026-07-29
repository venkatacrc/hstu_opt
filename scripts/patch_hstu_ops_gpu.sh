#!/usr/bin/env bash
# Patch installed hstu.hstu_ops_gpu so missing sm90 fake ops do not crash import.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

"${SCRIPT_DIR}/docker_run.sh" --name "${CONTAINER_NAME}-patch-hstu" -- \
  python3 "${CONTAINER_RAID}/hstu_opt/scripts/patch_hstu_ops_gpu.py"
