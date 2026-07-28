#!/usr/bin/env bash
# Diagnose / fix broken fbgemm_gpu_hstu installs.
#
# library.py loads:  hstu/fbgemm_gpu_experimental_hstu.so
# pip often ships:   hstu/fbgemm_gpu_experimental_hstu.cpython-312-*.so
# or, on a broken install, no .so at all (needs rebuild).
#
# Usage:
#   ./scripts/fixup_hstu_so.sh              # copy tagged->plain if possible
#   FORCE_REBUILD=1 ./scripts/fixup_hstu_so.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
  echo "==> FORCE_REBUILD=1 — wiping hstu package trees, then 03_install_train.sh"
  rm -rf \
    "${DEPS_ROOT}/lib/python3.12/site-packages/hstu" \
    "${DEPS_ROOT}/lib/python3.12/site-packages/"hstu*.egg-info \
    "${DEPS_ROOT}/lib/python3.12/site-packages/"fbgemm_gpu_hstu* \
    "${DEPS_ROOT}/local/lib/python3.12/dist-packages/hstu" \
    "${RECSYS_ROOT}/third_party/FBGEMM/fbgemm_gpu/experimental/hstu/build" \
    "${RECSYS_ROOT}/third_party/FBGEMM/fbgemm_gpu/experimental/hstu/dist" \
    2>/dev/null || true
  MAX_JOBS="${MAX_JOBS:-2}" "${SCRIPT_DIR}/03_install_train.sh"
  exit $?
fi

echo "==> Inspect / fix hstu .so inside container"
"${SCRIPT_DIR}/docker_run.sh" --name "${CONTAINER_NAME}-fixup-hstu" -- \
  bash -lc '
    set -euo pipefail
    python3 - <<'"'"'PY'"'"'
import glob, os, shutil, sys
pkg = None
for base in sys.path:
    d = os.path.join(base, "hstu")
    if (
        os.path.isdir(d)
        and os.path.isfile(os.path.join(d, "__init__.py"))
        and "examples/hstu" not in d
    ):
        pkg = d
        break
if not pkg:
    print("No installed hstu package on sys.path", file=sys.stderr)
    print("sys.path:", sys.path[:10], file=sys.stderr)
    raise SystemExit("Run: FORCE_REBUILD=1 ./scripts/fixup_hstu_so.sh")
print("package dir:", pkg)
print("contents:", sorted(os.listdir(pkg)))
plain = os.path.join(pkg, "fbgemm_gpu_experimental_hstu.so")
tagged = sorted(glob.glob(os.path.join(pkg, "fbgemm_gpu_experimental_hstu*.so")))
print("so candidates:", tagged)
if not tagged:
    raise SystemExit(
        "No CUDA .so in package — incomplete install.\n"
        "Run: FORCE_REBUILD=1 MAX_JOBS=2 ./scripts/fixup_hstu_so.sh"
    )
if not os.path.isfile(plain):
    src = [t for t in tagged if os.path.basename(t) != "fbgemm_gpu_experimental_hstu.so"][0]
    shutil.copy2(src, plain)
    print("copied", src, "->", plain)
from hstu import hstu_attn_varlen_func
import hstu
print("OK", hstu.__file__)
print("OK", hstu_attn_varlen_func)
PY
  '
