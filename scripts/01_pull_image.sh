#!/usr/bin/env bash
# Pull the NGC PyTorch image used for training-only HSTU installs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

require_cmd docker
print_env

echo "==> Pulling ${NGC_IMAGE}"
docker pull "${NGC_IMAGE}"
echo "==> Done. Image:"
docker images "${NGC_IMAGE}" --format '{{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}'
