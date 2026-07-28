#!/usr/bin/env bash
# Training-only dependency install inside the NGC PyTorch container.
# Skips Triton / FlexKV / NVE / inference AOTI from the official Dockerfile.
#
# Re-runnable: completed steps are skipped when imports already succeed.
# If fbgemm_gpu_hstu fails with ninja OOM, lower MAX_JOBS (default 4):
#   MAX_JOBS=2 ./scripts/03_install_train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SCRIPT_DIR}/00_env.sh"

require_cmd docker

if [[ ! -d "${RECSYS_ROOT}/examples/hstu" ]]; then
  echo "ERROR: recsys-examples not found at ${RECSYS_ROOT}. Run 02_clone_upstream.sh first." >&2
  exit 1
fi

print_env
echo "==> MAX_JOBS=${MAX_JOBS}  HSTU_ARCH_LIST=${HSTU_ARCH_LIST}"
TS="$(timestamp)"
LOG="${LOG_ROOT}/03_install_train_${TS}.log"
echo "==> Logging to ${LOG}"

# Install into the raid-mounted tree so packages persist across container restarts
# via a site-packages prefix under /raid/hstu/deps.
INSTALL_SCRIPT="${HSTU_RAID_ROOT}/_install_train_inner.sh"
cat > "${INSTALL_SCRIPT}" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail

RAID="${HSTU_CONTAINER_RAID:-/raid/hstu}"
RECSYS="${RAID}/recsys-examples"
DEPS="${RAID}/deps"
mkdir -p "${DEPS}/megatron-lm" "${DEPS}/wheels" "${DEPS}/logs"

export PYTHONUSERBASE="${DEPS}"
export PATH="${DEPS}/bin:${PATH}"
export PYTHONPATH="${DEPS}/lib/python3.12/site-packages:${DEPS}/lib/python3.11/site-packages:${RECSYS}/examples:${PYTHONPATH:-}"
# Avoid NGC pip constraint files interfering with source builds
unset PIP_CONSTRAINT || true

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export HSTU_ARCH_LIST="${HSTU_ARCH_LIST:-10.0}"
export HSTU_DISABLE_FP8="${HSTU_DISABLE_FP8:-TRUE}"
# Critical: default ninja uses all cores (~112) and OOMs compiling HSTU CUDA.
export MAX_JOBS="${MAX_JOBS:-4}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS}}"
# torch.utils.cpp_extension honors MAX_JOBS for ninja

py_ok() {
  local code="$1"
  python3 -c "${code}" >/dev/null 2>&1
}

echo "==> Python: $(python3 --version)  torch: $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "==> GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "==> MAX_JOBS=${MAX_JOBS} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} HSTU_ARCH_LIST=${HSTU_ARCH_LIST} HSTU_DISABLE_FP8=${HSTU_DISABLE_FP8}"

pip_install() {
  pip install --no-cache-dir --user "$@"
}

echo "==> [1/6] Python deps"
pip_install \
  gin-config \
  torchmetrics==1.0.3 \
  torchx \
  typing-extensions \
  iopath \
  pyvers \
  expiring_dict \
  cloudpickle \
  tensordict \
  orjson \
  setuptools-git-versioning \
  scikit-build \
  ninja \
  wheel

echo "==> [2/6] Megatron-Core core_v0.13.1"
if py_ok "import megatron.core"; then
  echo "    skip (megatron.core already importable)"
else
  if [[ ! -d "${DEPS}/megatron-lm/.git" ]]; then
    git clone -b core_v0.13.1 https://github.com/NVIDIA/Megatron-LM.git "${DEPS}/megatron-lm"
  fi
  pip install --no-deps --user -e "${DEPS}/megatron-lm"
fi

echo "==> [3/6] FBGEMM GPU (default/cuda) + TorchRec V1.5.0"
if py_ok "import fbgemm_gpu"; then
  echo "    skip fbgemm_gpu (already importable)"
else
  if [[ ! -d "${DEPS}/fbgemm/.git" ]]; then
    git clone --recursive -b v1.5.0 https://github.com/pytorch/FBGEMM.git "${DEPS}/fbgemm"
  fi
  pushd "${DEPS}/fbgemm/fbgemm_gpu" >/dev/null
  # Limit parallelism for the large CUDA build
  MAX_JOBS="${MAX_JOBS}" \
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  python3 setup.py install --prefix="${DEPS}" --build-target=default --build-variant=cuda \
    -DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
  popd >/dev/null
fi

if py_ok "import torchrec"; then
  echo "    skip torchrec (already importable)"
else
  if [[ ! -d "${DEPS}/torchrec/.git" ]]; then
    git clone --recursive -b release/V1.5.0 https://github.com/pytorch/torchrec.git "${DEPS}/torchrec"
  fi
  pip install --no-deps --user -e "${DEPS}/torchrec"
fi

echo "==> [4/6] fbgemm_gpu_hstu (import hstu) from recsys submodule"
if py_ok "import hstu"; then
  echo "    skip hstu (already importable)"
else
  if [[ ! -d "${RECSYS}/third_party/FBGEMM/fbgemm_gpu/experimental/hstu" ]]; then
    echo "ERROR: missing third_party/FBGEMM HSTU sources; re-run 02_clone_upstream.sh" >&2
    exit 1
  fi
  pushd "${RECSYS}/third_party/FBGEMM/fbgemm_gpu/experimental/hstu" >/dev/null
  # Clean prior failed build artifacts that can confuse ninja
  rm -rf build dist ./*.egg-info ./fbgemm_gpu_hstu.egg-info 2>/dev/null || true

  # B200 bf16 path: build sm_100 only, skip FP8 (your log failed in sm90 e4m3 bwd).
  # MAX_JOBS must stay low — override with MAX_JOBS=2 if OOM persists.
  export HSTU_DISABLE_86OR89=TRUE
  export HSTU_DISABLE_ARBITRARY=TRUE
  export HSTU_DISABLE_LOCAL=TRUE
  export HSTU_DISABLE_RAB=TRUE
  export HSTU_DISABLE_DRAB=TRUE
  export HSTU_DISABLE_FP16=TRUE
  export HSTU_DISABLE_FP8="${HSTU_DISABLE_FP8:-TRUE}"
  export HSTU_DISABLE_120=TRUE
  export HSTU_ARCH_LIST
  export MAX_JOBS
  export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"

  echo "    building fbgemm_gpu_hstu with MAX_JOBS=${MAX_JOBS} HSTU_ARCH_LIST=${HSTU_ARCH_LIST} HSTU_DISABLE_FP8=${HSTU_DISABLE_FP8}"
  HSTU_LOG="${DEPS}/logs/fbgemm_gpu_hstu_build.log"
  set +e
  MAX_JOBS="${MAX_JOBS}" \
  CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}" \
  HSTU_ARCH_LIST="${HSTU_ARCH_LIST}" \
  HSTU_DISABLE_86OR89=TRUE \
  HSTU_DISABLE_ARBITRARY=TRUE \
  HSTU_DISABLE_LOCAL=TRUE \
  HSTU_DISABLE_RAB=TRUE \
  HSTU_DISABLE_DRAB=TRUE \
  HSTU_DISABLE_FP16=TRUE \
  HSTU_DISABLE_FP8="${HSTU_DISABLE_FP8}" \
  HSTU_DISABLE_120=TRUE \
  pip install --no-build-isolation --no-cache-dir --user . 2>&1 | tee "${HSTU_LOG}"
  HSTU_RC=${PIPESTATUS[0]}
  set -e
  if [[ ${HSTU_RC} -ne 0 ]]; then
    echo "ERROR: fbgemm_gpu_hstu build failed (exit ${HSTU_RC})." >&2
    echo "---- last 80 lines of ${HSTU_LOG} ----" >&2
    tail -n 80 "${HSTU_LOG}" >&2 || true
    echo "---- tip: retry with MAX_JOBS=2 ./scripts/03_install_train.sh ----" >&2
    exit "${HSTU_RC}"
  fi
  popd >/dev/null
fi

echo "==> [5/6] DynamicEmb"
if py_ok "import dynamicemb"; then
  echo "    skip dynamicemb (already importable)"
else
  pushd "${RECSYS}/corelib/dynamicemb" >/dev/null
  MAX_JOBS="${MAX_JOBS}" \
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  python3 setup.py install --prefix="${DEPS}"
  popd >/dev/null
fi

echo "==> [6/6] examples/commons custom CUDA ops"
# commons installs as a collection of extensions; always rebuild if import of a
# known helper fails. Best-effort: run install when marker missing.
COMMONS_MARKER="${DEPS}/.commons_installed"
if [[ -f "${COMMONS_MARKER}" ]]; then
  echo "    skip commons (marker ${COMMONS_MARKER})"
else
  pushd "${RECSYS}/examples/commons" >/dev/null
  MAX_JOBS="${MAX_JOBS}" \
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  python3 setup.py install --prefix="${DEPS}"
  popd >/dev/null
  touch "${COMMONS_MARKER}"
fi

echo "==> Smoke imports"
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
import gin; print("gin ok")
import megatron.core; print("megatron.core ok")
import torchrec; print("torchrec", getattr(torchrec, "__version__", "?"))
import fbgemm_gpu; print("fbgemm_gpu ok")
import hstu; print("hstu (fbgemm_gpu_hstu) ok")
import dynamicemb; print("dynamicemb ok")
print("INSTALL_OK")
PY

echo "==> Install complete. PYTHONUSERBASE=${DEPS} MAX_JOBS=${MAX_JOBS}"
INNER

chmod +x "${INSTALL_SCRIPT}"

# Persist user site under /raid/hstu/deps across runs
"${SCRIPT_DIR}/docker_run.sh" --privileged --name "${CONTAINER_NAME}-install" -- \
  bash -lc "
    export HSTU_CONTAINER_RAID=${CONTAINER_RAID}
    export TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}'
    export HSTU_ARCH_LIST='${HSTU_ARCH_LIST}'
    export HSTU_DISABLE_FP8='${HSTU_DISABLE_FP8}'
    export MAX_JOBS='${MAX_JOBS}'
    export CMAKE_BUILD_PARALLEL_LEVEL='${MAX_JOBS}'
    export PYTHONUSERBASE=${CONTAINER_RAID}/deps
    export PATH=${CONTAINER_RAID}/deps/bin:\$PATH
    export PYTHONPATH=${CONTAINER_RAID}/deps/lib/python3.12/site-packages:${CONTAINER_RAID}/deps/lib/python3.11/site-packages:${CONTAINER_RAID}/recsys-examples/examples:\$PYTHONPATH
    bash ${CONTAINER_RAID}/_install_train_inner.sh
  " 2>&1 | tee "${LOG}"

echo "==> Install log: ${LOG}"
echo "==> If smoke printed INSTALL_OK, proceed to 04_preprocess.sh"
echo "==> On fbgemm_gpu_hstu OOM/ninja failure: MAX_JOBS=2 ./scripts/03_install_train.sh"
