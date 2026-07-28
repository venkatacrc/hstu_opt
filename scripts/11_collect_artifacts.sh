#!/usr/bin/env bash
# Collect logs + text summaries of nsys/ncu reports for paste-back.
# Binary .nsys-rep / .ncu-rep stay on /raid; a tarball of text artifacts is produced.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

TS="$(timestamp)"
HOST="$(hostname -s)"
SUM_DIR="${PROFILE_ROOT}/summaries/${TS}_${HOST}"
mkdir -p "${SUM_DIR}"

print_env
echo "==> Collecting into ${SUM_DIR}"

{
  echo "host=$(hostname)"
  echo "date=$(date -Is)"
  echo "uname=$(uname -a)"
  echo "nsys=$(${NSYS_BIN:-nsys} --version 2>&1 | head -n 1 || true)"
  echo "ncu=$(${NCU_BIN:-ncu} --version 2>&1 | head -n 3 || true)"
  echo "---- nvidia-smi ----"
  nvidia-smi || true
  echo "---- recsys sha ----"
  cat "${HSTU_RAID_ROOT}/recsys_sha.txt" 2>/dev/null || true
  git -C "${RECSYS_ROOT}" log -1 --oneline 2>/dev/null || true
  echo "---- hstu_opt HEAD ----"
  git -C "${HSTU_OPT_ROOT}" log -1 --oneline 2>/dev/null || true
} > "${SUM_DIR}/environment.txt"

# Copy recent logs (last 30)
mkdir -p "${SUM_DIR}/logs"
find "${LOG_ROOT}" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -n 30 | awk '{print $2}' \
  | while read -r f; do cp -a "$f" "${SUM_DIR}/logs/" || true; done
# macOS find lacks -printf; fallback
if [[ ! -d "${SUM_DIR}/logs" ]] || [[ -z "$(ls -A "${SUM_DIR}/logs" 2>/dev/null || true)" ]]; then
  ls -1t "${LOG_ROOT}"/*.log 2>/dev/null | head -n 30 | while read -r f; do
    cp -a "$f" "${SUM_DIR}/logs/" || true
  done
fi

# nsys stats for each report
mkdir -p "${SUM_DIR}/nsys"
shopt -s nullglob
for rep in "${PROFILE_ROOT}/nsys"/*.nsys-rep "${PROFILE_ROOT}/nsys"/*.qdrep; do
  base="$(basename "${rep}")"
  echo "==> nsys stats ${base}"
  if [[ -n "${NSYS_BIN}" ]]; then
    "${NSYS_BIN}" stats \
      --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
      --format csv,column \
      -o "${SUM_DIR}/nsys/${base}" \
      "${rep}" > "${SUM_DIR}/nsys/${base}.stats.txt" 2>&1 || \
      "${NSYS_BIN}" stats "${rep}" > "${SUM_DIR}/nsys/${base}.stats.txt" 2>&1 || true
  fi
  ls -lah "${rep}" >> "${SUM_DIR}/nsys/files.txt"
done

# ncu text export
mkdir -p "${SUM_DIR}/ncu"
for rep in "${PROFILE_ROOT}/ncu"/*.ncu-rep; do
  base="$(basename "${rep}" .ncu-rep)"
  echo "==> ncu import ${base}"
  if [[ -n "${NCU_BIN}" ]]; then
    "${NCU_BIN}" --import "${rep}" --page details \
      > "${SUM_DIR}/ncu/${base}.details.txt" 2>&1 || true
    "${NCU_BIN}" --import "${rep}" --page raw \
      > "${SUM_DIR}/ncu/${base}.raw.txt" 2>&1 || true
  fi
  ls -lah "${rep}" >> "${SUM_DIR}/ncu/files.txt"
done
shopt -u nullglob

# Tail of most recent train/profile logs for quick paste
{
  echo "######## recent log tails ########"
  ls -1t "${LOG_ROOT}"/*.log 2>/dev/null | head -n 8 | while read -r f; do
    echo "===== $(basename "$f") ====="
    tail -n 60 "$f" || true
    echo
  done
} > "${SUM_DIR}/PASTE_ME.txt"

TAR="${PROFILE_ROOT}/summaries/hstu_profiles_${TS}_${HOST}.tar.gz"
tar -C "${PROFILE_ROOT}/summaries" -czf "${TAR}" "${TS}_${HOST}"

echo
echo "==> Summary dir: ${SUM_DIR}"
echo "==> Tarball:     ${TAR}"
echo "==> Paste-back:  ${SUM_DIR}/PASTE_ME.txt  (+ environment.txt if useful)"
echo "==> Binary reports remain under ${PROFILE_ROOT}/{nsys,ncu}/"
cat "${SUM_DIR}/PASTE_ME.txt"
