#!/usr/bin/env bash
# Quick fix: if the tagged CUDA extension .so exists but the plain name that
# hstu/library.py loads does not, copy it. Also reports package layout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

"${SCRIPT_DIR}/docker_run.sh" -- \
  python3 - <<'PY'
import glob, os, shutil, importlib.util

spec = importlib.util.find_spec("hstu")
if spec is None or not spec.origin:
    raise SystemExit("hstu not installed — run MAX_JOBS=2 ./scripts/03_install_train.sh")

pkg = os.path.dirname(spec.origin)
print("hstu package:", pkg)
print("contents:")
for name in sorted(os.listdir(pkg)):
    path = os.path.join(pkg, name)
    if name.endswith(".so") or name.endswith(".py"):
        print(f"  {name}  ({os.path.getsize(path)} bytes)")

plain = os.path.join(pkg, "fbgemm_gpu_experimental_hstu.so")
tagged = sorted(
    p for p in glob.glob(os.path.join(pkg, "fbgemm_gpu_experimental_hstu*.so"))
    if os.path.basename(p) != "fbgemm_gpu_experimental_hstu.so"
)
if os.path.exists(plain):
    print("plain .so already present:", plain)
elif tagged:
    shutil.copy2(tagged[0], plain)
    print(f"copied {tagged[0]} -> {plain}")
else:
    raise SystemExit(
        "No CUDA extension .so found under the hstu package. "
        "Rebuild with: MAX_JOBS=2 ./scripts/03_install_train.sh"
    )

from hstu import hstu_attn_varlen_func
print("import ok:", hstu_attn_varlen_func)
print("FIX_OK")
PY
