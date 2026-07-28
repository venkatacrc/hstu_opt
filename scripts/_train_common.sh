#!/usr/bin/env bash
# Shared helpers for train / profile launchers. Sourced by 05–10 scripts.

# shellcheck disable=SC1091
: "${SCRIPT_DIR:?SCRIPT_DIR must be set before sourcing _train_common.sh}"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

gin_in_container() {
  # Map host gin path under HSTU_OPT_ROOT to container path
  local host_gin="$1"
  echo "${CONTAINER_RAID}/hstu_opt/${host_gin#${HSTU_OPT_ROOT}/}"
}

run_torchrun() {
  local nproc="$1"
  local entry="$2"
  local gin_host="$3"
  shift 3
  local gin_ctr
  gin_ctr="$(gin_in_container "${gin_host}")"

  if [[ ! -f "${gin_host}" ]]; then
    echo "ERROR: gin config not found: ${gin_host}" >&2
    exit 1
  fi
  if [[ ! -f "${HSTU_EXAMPLE_ROOT}/${entry}" ]]; then
    echo "ERROR: training entry not found: ${HSTU_EXAMPLE_ROOT}/${entry}" >&2
    exit 1
  fi

  local extra_docker=()
  local privileged=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --privileged) privileged=1; shift ;;
      --gpus) extra_docker+=(--gpus "$2"); shift 2 ;;
      *) break ;;
    esac
  done

  local docker_cmd=("${SCRIPT_DIR}/docker_run.sh")
  if [[ ${privileged} -eq 1 ]]; then
    docker_cmd+=(--privileged)
  fi
  if [[ ${#extra_docker[@]} -gt 0 ]]; then
    docker_cmd+=("${extra_docker[@]}")
  fi

  "${docker_cmd[@]}" -- \
    bash -lc "
      set -euo pipefail
      cd ${CONTAINER_WORKDIR}
      export PYTHONPATH=${CONTAINER_RAID}/recsys-examples/examples:\${PYTHONPATH:-}
      export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}
      export FILL_DYNAMICEMB_TABLES=1
      torchrun \
        --standalone \
        --nproc_per_node=${nproc} \
        --master_port=${MASTER_PORT} \
        ${entry} \
        --gin-config-file ${gin_ctr}
    "
}
