#!/usr/bin/env bash
# 8-GPU MovieLens-20M ranking smoke / train (no nsys/ncu).
#
# Usage:
#   ./05_train_ranking.sh           # smoke gin (50 iters)
#   ./05_train_ranking.sh profile   # profile gin without wrapping profiler
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_train_common.sh
source "${SCRIPT_DIR}/_train_common.sh"

MODE="${1:-smoke}"
case "${MODE}" in
  smoke) GIN="${RANKING_SMOKE_GIN}" ;;
  profile) GIN="${RANKING_PROFILE_GIN}" ;;
  *)
    echo "Usage: $0 [smoke|profile]" >&2
    exit 2
    ;;
esac

TS="$(timestamp)"
LOG="${LOG_ROOT}/05_train_ranking_${MODE}_${TS}.log"
print_env
echo "==> Ranking train mode=${MODE} nproc=${NPROC}"
echo "==> Gin: ${GIN}"
echo "==> Log: ${LOG}"

{
  echo "==== host nvidia-smi ===="
  nvidia-smi || true
  echo "==== start $(date -Is) ===="
  run_torchrun "${NPROC}" "training/pretrain_gr_ranking.py" "${GIN}"
  echo "==== end $(date -Is) ===="
} 2>&1 | tee "${LOG}"

echo "==> Done. Log: ${LOG}"
echo "Paste the last ~80 lines of the log if the run failed."
