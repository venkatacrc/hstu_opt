#!/usr/bin/env bash
# Host ncu profile of MovieLens-20M retrieval on a single GPU (rank 0).
#
# Optional env:
#   NCU_SET=full|detailed|basic
#   NCU_KERNEL_REGEX=regex
#   NCU_REPLAY_MODE=application|kernel
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_train_common.sh
source "${SCRIPT_DIR}/_train_common.sh"

require_cmd docker
if [[ -z "${NCU_BIN}" ]]; then
  echo "ERROR: ncu not found on PATH. Install Nsight Compute or set NCU_BIN." >&2
  exit 1
fi

NCU_SET="${NCU_SET:-detailed}"
NCU_REPLAY_MODE="${NCU_REPLAY_MODE:-application}"
TS="$(timestamp)"
HOST="$(hostname -s)"
OUT="${PROFILE_ROOT}/ncu/retrieval_${TS}_${HOST}"
LOG="${LOG_ROOT}/10_ncu_retrieval_${TS}.log"
GIN_CTR="$(gin_in_container "${RETRIEVAL_PROFILE_GIN}")"

case "${NCU_SET}" in
  full) NCU_SET_ARGS=(--set full) ;;
  basic) NCU_SET_ARGS=(--set basic) ;;
  detailed)
    NCU_SET_ARGS=(
      --section LaunchStats
      --section Occupancy
      --section SpeedOfLight
      --section MemoryWorkloadAnalysis
      --section ComputeWorkloadAnalysis
    )
    ;;
  *)
    echo "ERROR: unknown NCU_SET=${NCU_SET}" >&2
    exit 2
    ;;
esac

NCU_FILTER_ARGS=()
if [[ -n "${NCU_KERNEL_REGEX:-}" ]]; then
  NCU_FILTER_ARGS+=(--kernel-name-base-expr "${NCU_KERNEL_REGEX}")
fi

print_env
echo "==> ncu: ${NCU_BIN}"
"${NCU_BIN}" --version || true
echo "==> Output prefix: ${OUT}"
echo "==> Log: ${LOG}"

{
  echo "==== start $(date -Is) ===="
  "${NCU_BIN}" \
    --target-processes all \
    --replay-mode "${NCU_REPLAY_MODE}" \
    --profile-from-start off \
    "${NCU_SET_ARGS[@]}" \
    "${NCU_FILTER_ARGS[@]}" \
    -o "${OUT}" \
    -f \
    -- "${SCRIPT_DIR}/docker_run.sh" --privileged --gpus device=0 --name "${CONTAINER_NAME}-ncu-retr" -- \
      bash -lc "
        set -euo pipefail
        cd ${CONTAINER_WORKDIR}
        export CUDA_VISIBLE_DEVICES=0
        export PYTHONPATH=${CONTAINER_WORKDIR}:${CONTAINER_RAID}/deps/lib/python3.12/site-packages:${CONTAINER_RAID}/deps/local/lib/python3.12/dist-packages:${CONTAINER_RAID}/recsys-examples/examples
        export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}
        export FILL_DYNAMICEMB_TABLES=1
        torchrun \
          --standalone \
          --nproc_per_node=1 \
          --master_port=${MASTER_PORT} \
          training/pretrain_gr_retrieval.py \
          --gin-config-file ${GIN_CTR}
      "
  echo "==== end $(date -Is) ===="
  ls -lah "${OUT}".* || true
} 2>&1 | tee "${LOG}"

echo "==> ncu report: ${OUT}.ncu-rep"
echo "==> Log: ${LOG}"
