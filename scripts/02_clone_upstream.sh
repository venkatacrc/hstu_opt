#!/usr/bin/env bash
# Clone NVIDIA/recsys-examples (recursive) onto /raid.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

require_cmd git
print_env

if [[ -d "${RECSYS_ROOT}/.git" ]]; then
  echo "==> Existing clone at ${RECSYS_ROOT}"
  git -C "${RECSYS_ROOT}" fetch --tags origin
  git -C "${RECSYS_ROOT}" checkout "${RECSYS_REF}"
  # If RECSYS_REF is a branch name, fast-forward; ignore for detached SHAs/tags.
  git -C "${RECSYS_ROOT}" pull --ff-only origin "${RECSYS_REF}" 2>/dev/null || true
else
  echo "==> Cloning ${RECSYS_REPO} @ ${RECSYS_REF} -> ${RECSYS_ROOT}"
  mkdir -p "$(dirname "${RECSYS_ROOT}")"
  git clone --recursive --branch "${RECSYS_REF}" "${RECSYS_REPO}" "${RECSYS_ROOT}" \
    || git clone --recursive "${RECSYS_REPO}" "${RECSYS_ROOT}"
  if [[ "${RECSYS_REF}" != "main" ]]; then
    git -C "${RECSYS_ROOT}" checkout "${RECSYS_REF}"
  fi
fi

echo "==> Updating submodules (FBGEMM, etc.)"
git -C "${RECSYS_ROOT}" submodule update --init --recursive

echo "==> Upstream HEAD:"
git -C "${RECSYS_ROOT}" rev-parse HEAD
git -C "${RECSYS_ROOT}" log -1 --oneline
echo "${RECSYS_REF}" > "${HSTU_RAID_ROOT}/recsys_ref.txt"
git -C "${RECSYS_ROOT}" rev-parse HEAD > "${HSTU_RAID_ROOT}/recsys_sha.txt"
echo "==> Done."
