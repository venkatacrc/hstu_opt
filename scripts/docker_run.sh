#!/usr/bin/env bash
# Common docker run helper for training / profiling.
#
# Usage:
#   ./docker_run.sh [--privileged] [--gpus GPU_SPEC] [--name NAME] -- <cmd...>
#   ./docker_run.sh -- python -c 'import torch; print(torch.cuda.device_count())'
#
# Env (from 00_env.sh):
#   NGC_IMAGE, HSTU_RAID_ROOT, CONTAINER_RAID, CONTAINER_WORKDIR, CONTAINER_NAME

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

require_cmd docker

PRIVILEGED=0
GPU_SPEC="all"
NAME="${CONTAINER_NAME}"
EXTRA_DOCKER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --privileged)
      PRIVILEGED=1
      shift
      ;;
    --gpus)
      GPU_SPEC="$2"
      shift 2
      ;;
    --gpus=*)
      GPU_SPEC="${1#*=}"
      shift
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    --name=*)
      NAME="${1#*=}"
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      EXTRA_DOCKER_ARGS+=("$1")
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [--privileged] [--gpus all|device=0] [--name NAME] -- <command>" >&2
  exit 2
fi

# Persistent pip --user prefix from 03_install_train.sh.
# setup.py --prefix also used deps/local/lib/.../dist-packages — keep both.
DEPS_IN_CTR="${CONTAINER_RAID}/deps"
SITE_PATH="\
${DEPS_IN_CTR}/lib/python3.12/site-packages:\
${DEPS_IN_CTR}/local/lib/python3.12/dist-packages:\
${DEPS_IN_CTR}/lib/python3.12/dist-packages:\
${DEPS_IN_CTR}/local/lib/python3.12/site-packages:\
${DEPS_IN_CTR}/lib/python3.11/site-packages"

DOCKER_ARGS=(
  --rm
  --gpus "${GPU_SPEC}"
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --network host
  --name "${NAME}-$$"
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
  -e CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS}"
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
  -e HSTU_ARCH_LIST="${HSTU_ARCH_LIST}"
  -e MAX_JOBS="${MAX_JOBS:-4}"
  -e CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS:-4}}"
  -e PYTHONUSERBASE="${DEPS_IN_CTR}"
  # Path order matters:
  #  1) examples/hstu — upstream `configs`, `model`, `modules`, ...
  #  2) SITE_PATH — installed `hstu` (fbgemm_gpu_hstu), dynamicemb, ...
  #  3) examples/ — `commons` (must come AFTER site-packages: the
  #     examples/hstu directory would otherwise shadow the `hstu` package)
  -e PYTHONPATH="${CONTAINER_WORKDIR}:${SITE_PATH}:${CONTAINER_RAID}/recsys-examples/examples"
  -e PATH="${DEPS_IN_CTR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  -e FILL_DYNAMICEMB_TABLES="${FILL_DYNAMICEMB_TABLES:-1}"
  -e HSTU_FORCE_NATIVE="${HSTU_FORCE_NATIVE:-1}"
  -v "${HSTU_RAID_ROOT}:${CONTAINER_RAID}"
  -v "${HSTU_OPT_ROOT}:${CONTAINER_RAID}/hstu_opt:ro"
  -w "${CONTAINER_WORKDIR}"
)

if [[ ${PRIVILEGED} -eq 1 ]]; then
  DOCKER_ARGS+=(--privileged --cap-add=SYS_ADMIN)
fi

if [[ ${#EXTRA_DOCKER_ARGS[@]} -gt 0 ]]; then
  DOCKER_ARGS+=("${EXTRA_DOCKER_ARGS[@]}")
fi

exec docker run "${DOCKER_ARGS[@]}" "${NGC_IMAGE}" "$@"
