#!/usr/bin/env bash
# Convenience: run one-time setup steps 01→04 sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

print_env
"${SCRIPT_DIR}/01_pull_image.sh"
"${SCRIPT_DIR}/02_clone_upstream.sh"
"${SCRIPT_DIR}/03_install_train.sh"
"${SCRIPT_DIR}/04_preprocess.sh"
echo "==> Setup complete. Next: ./scripts/05_train_ranking.sh smoke && ./scripts/06_train_retrieval.sh smoke"
