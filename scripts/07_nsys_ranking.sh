#!/usr/bin/env bash
# Host nsys profile of 8-GPU MovieLens-20M ranking (cudaProfilerApi window).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_train_common.sh
source "${SCRIPT_DIR}/_train_common.sh"

require_cmd docker
if [[ -z "${NSYS_BIN}" ]]; then
  echo "ERROR: nsys not found on PATH. Install Nsight Systems or set NSYS_BIN." >&2
  exit 1
fi

TS="$(timestamp)"
HOST="$(hostname -s)"
OUT="${PROFILE_ROOT}/nsys/ranking_${TS}_${HOST}"
LOG="${LOG_ROOT}/07_nsys_ranking_${TS}.log"
GIN_CTR="$(gin_in_container "${RANKING_PROFILE_GIN}")"

print_env
echo "==> nsys: ${NSYS_BIN}"
"${NSYS_BIN}" --version || true
echo "==> Output prefix: ${OUT}"
echo "==> Log: ${LOG}"

# Prefer host nsys wrapping a privileged container so CUPTI can attach (see upstream #397).
{
  echo "==== start $(date -Is) ===="
  CUBLAS_NVTX_LEVEL=2 \
  "${NSYS_BIN}" profile \
    -o "${OUT}" \
    -f true \
    -s none \
    -t cuda,nvtx,cublas \
    -c cudaProfilerApi \
    --cpuctxsw none \
    --cuda-flush-interval 100 \
    --capture-range-end=stop \
    --cuda-graph-trace=node \
    --trace-fork-before-exec=true \
    -- "${SCRIPT_DIR}/docker_run.sh" --privileged --name "${CONTAINER_NAME}-nsys-rank" -- \
      bash -lc "
        set -euo pipefail
        cd ${CONTAINER_WORKDIR}
        export PYTHONPATH=${CONTAINER_WORKDIR}:${CONTAINER_RAID}/recsys-examples/examples:\${PYTHONPATH:-}
        export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}
        export FILL_DYNAMICEMB_TABLES=1
        export CUBLAS_NVTX_LEVEL=2
        torchrun \
          --standalone \
          --nproc_per_node=${NPROC} \
          --master_port=${MASTER_PORT} \
          training/pretrain_gr_ranking.py \
          --gin-config-file ${GIN_CTR}
      "
  echo "==== end $(date -Is) ===="
  ls -lah "${OUT}".* || true
} 2>&1 | tee "${LOG}"

echo "==> nsys report: ${OUT}.nsys-rep (or .qdrep)"
echo "==> Log: ${LOG}"
