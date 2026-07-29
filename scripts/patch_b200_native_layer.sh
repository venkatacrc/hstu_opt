#!/usr/bin/env bash
# Force NATIVE HSTU layer on B200 so MovieLens contextual tokens work.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

UTILS="${HSTU_EXAMPLE_ROOT}/training/trainer/utils.py"
if [[ ! -f "${UTILS}" ]]; then
  echo "ERROR: ${UTILS} not found. Run 02_clone_upstream.sh first." >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/patch_b200_native_layer.py" "${UTILS}"
echo "==> Default HSTU_FORCE_NATIVE=1 (set to 0 to use FUSED when no contextual tokens)"
