#!/usr/bin/env bash
# Shared environment for HSTU 8xB200 training + nsys/ncu profiling.
# Usage: source "$(dirname "$0")/00_env.sh"

# Do not use `set -e` here — this file is sourced.

export HSTU_RAID_ROOT="${HSTU_RAID_ROOT:-/raid/hstu}"
export HSTU_OPT_ROOT="${HSTU_OPT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export RECSYS_ROOT="${RECSYS_ROOT:-${HSTU_RAID_ROOT}/recsys-examples}"
export HSTU_EXAMPLE_ROOT="${HSTU_EXAMPLE_ROOT:-${RECSYS_ROOT}/examples/hstu}"
export COMMONS_ROOT="${COMMONS_ROOT:-${RECSYS_ROOT}/examples/commons}"

export DATA_ROOT="${DATA_ROOT:-${HSTU_RAID_ROOT}/data}"
export LOG_ROOT="${LOG_ROOT:-${HSTU_RAID_ROOT}/logs}"
export PROFILE_ROOT="${PROFILE_ROOT:-${HSTU_RAID_ROOT}/profiles}"
export DEPS_ROOT="${DEPS_ROOT:-${HSTU_RAID_ROOT}/deps}"

export NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/pytorch:26.05-py3}"
export CONTAINER_NAME="${CONTAINER_NAME:-hstu-train}"

export NPROC="${NPROC:-8}"
export MASTER_PORT="${MASTER_PORT:-6000}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"

# Upstream pin (override with RECSYS_REF=main|tag|sha)
export RECSYS_REPO="${RECSYS_REPO:-https://github.com/NVIDIA/recsys-examples.git}"
export RECSYS_REF="${RECSYS_REF:-main}"

# Host profilers (already installed on b200-50)
export NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
export NCU_BIN="${NCU_BIN:-$(command -v ncu || true)}"

# CUDA arch list for Blackwell B200 (sm_100) + forward-compat
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5 8.0 8.6 9.0 10.0 12.0}"
export HSTU_ARCH_LIST="${HSTU_ARCH_LIST:-8.0 9.0 10.0 12.0}"

# Gin configs shipped in this repo
export RANKING_PROFILE_GIN="${RANKING_PROFILE_GIN:-${HSTU_OPT_ROOT}/configs/movielen_ranking_profile.gin}"
export RETRIEVAL_PROFILE_GIN="${RETRIEVAL_PROFILE_GIN:-${HSTU_OPT_ROOT}/configs/movielen_retrieval_profile.gin}"
export RANKING_SMOKE_GIN="${RANKING_SMOKE_GIN:-${HSTU_OPT_ROOT}/configs/movielen_ranking_smoke.gin}"
export RETRIEVAL_SMOKE_GIN="${RETRIEVAL_SMOKE_GIN:-${HSTU_OPT_ROOT}/configs/movielen_retrieval_smoke.gin}"

# Inside-container mount of the raid tree
export CONTAINER_RAID="${CONTAINER_RAID:-/raid/hstu}"
export CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-${CONTAINER_RAID}/recsys-examples/examples/hstu}"

mkdir -p \
  "${HSTU_RAID_ROOT}" \
  "${DATA_ROOT}" \
  "${LOG_ROOT}" \
  "${PROFILE_ROOT}/nsys" \
  "${PROFILE_ROOT}/ncu" \
  "${PROFILE_ROOT}/summaries" \
  "${DEPS_ROOT}"

timestamp() {
  date +%Y%m%d_%H%M%S
}

require_cmd() {
  local c="$1"
  if ! command -v "${c}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${c}" >&2
    exit 1
  fi
}

print_env() {
  cat <<EOF
HSTU env
  HSTU_OPT_ROOT=${HSTU_OPT_ROOT}
  HSTU_RAID_ROOT=${HSTU_RAID_ROOT}
  RECSYS_ROOT=${RECSYS_ROOT}
  DATA_ROOT=${DATA_ROOT}
  LOG_ROOT=${LOG_ROOT}
  PROFILE_ROOT=${PROFILE_ROOT}
  NGC_IMAGE=${NGC_IMAGE}
  NPROC=${NPROC}
  RECSYS_REF=${RECSYS_REF}
  NSYS_BIN=${NSYS_BIN:-<missing>}
  NCU_BIN=${NCU_BIN:-<missing>}
EOF
}
