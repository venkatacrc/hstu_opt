#!/usr/bin/env bash
# Download + preprocess MovieLens-20M into /raid/hstu/data and link into upstream.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

DATASET_NAME="${DATASET_NAME:-ml-20m}"
TS="$(timestamp)"
LOG="${LOG_ROOT}/04_preprocess_${DATASET_NAME}_${TS}.log"

if [[ ! -d "${COMMONS_ROOT}" ]]; then
  echo "ERROR: ${COMMONS_ROOT} missing. Run 02_clone_upstream.sh first." >&2
  exit 1
fi

print_env
echo "==> Preprocessing ${DATASET_NAME} -> ${DATA_ROOT}"
echo "==> Log: ${LOG}"

mkdir -p "${DATA_ROOT}"

"${SCRIPT_DIR}/docker_run.sh" --name "${CONTAINER_NAME}-preprocess" -- \
  bash -lc "
    set -euo pipefail
    cd ${CONTAINER_RAID}/recsys-examples/examples/commons
    mkdir -p ${CONTAINER_RAID}/data
    python3 ./hstu_data_preprocessor.py \
      --dataset_name ${DATASET_NAME} \
      --dataset_path ${CONTAINER_RAID}/data \
      --training
  " 2>&1 | tee "${LOG}"

# Make data discoverable from both commons and hstu working dirs
mkdir -p "${COMMONS_ROOT}/tmp_data" "${HSTU_EXAMPLE_ROOT}/tmp_data"
# Prefer symlink of processed prefix into the default tmp_data locations
if [[ -d "${DATA_ROOT}/${DATASET_NAME}" ]]; then
  ln -sfn "${DATA_ROOT}/${DATASET_NAME}" "${COMMONS_ROOT}/tmp_data/${DATASET_NAME}"
  ln -sfn "${DATA_ROOT}/${DATASET_NAME}" "${HSTU_EXAMPLE_ROOT}/tmp_data/${DATASET_NAME}"
fi
# Also expose entire data root as tmp_data content when processor writes flat files there
ln -sfn "${DATA_ROOT}" "${COMMONS_ROOT}/tmp_data_raid"
ln -sfn "${DATA_ROOT}" "${HSTU_EXAMPLE_ROOT}/tmp_data_raid"

echo "==> Data listing:"
ls -lah "${DATA_ROOT}" | tee -a "${LOG}"
echo "==> Done. Proceed to 05_train_ranking.sh / 06_train_retrieval.sh"
