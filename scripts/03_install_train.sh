#!/usr/bin/env bash
# Training-only dependency install inside the NGC PyTorch container.
# Skips Triton / FlexKV / NVE / inference AOTI from the official Dockerfile.
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
SITE="${DEPS}/site-packages"
mkdir -p "${SITE}" "${DEPS}/megatron-lm" "${DEPS}/wheels"

export PYTHONPATH="${SITE}:${RECSYS}/examples:${PYTHONPATH:-}"
export PIP_CONSTRAINT="${PIP_CONSTRAINT:-}"
# Prefer writing installs into our persistent prefix
export PYTHONUSERBASE="${DEPS}"
export PATH="${DEPS}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5 8.0 8.6 9.0 10.0 12.0}"
export HSTU_ARCH_LIST="${HSTU_ARCH_LIST:-8.0 9.0 10.0 12.0}"

echo "==> Python: $(python3 --version)  torch: $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "==> GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"

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
  scikit-build

echo "==> [2/6] Megatron-Core core_v0.13.1"
if [[ ! -d "${DEPS}/megatron-lm/.git" ]]; then
  git clone -b core_v0.13.1 https://github.com/NVIDIA/Megatron-LM.git "${DEPS}/megatron-lm"
fi
pip install --no-deps --user -e "${DEPS}/megatron-lm"

echo "==> [3/6] FBGEMM GPU (default/cuda) + TorchRec V1.5.0"
if [[ ! -d "${DEPS}/fbgemm/.git" ]]; then
  git clone --recursive -b v1.5.0 https://github.com/pytorch/FBGEMM.git "${DEPS}/fbgemm"
fi
pushd "${DEPS}/fbgemm/fbgemm_gpu" >/dev/null
python3 setup.py install --prefix="${DEPS}" --build-target=default --build-variant=cuda \
  -DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
popd >/dev/null

if [[ ! -d "${DEPS}/torchrec/.git" ]]; then
  git clone --recursive -b release/V1.5.0 https://github.com/pytorch/torchrec.git "${DEPS}/torchrec"
fi
pip install --no-deps --user -e "${DEPS}/torchrec"

echo "==> [4/6] fbgemm_gpu_hstu (import hstu) from recsys submodule"
if [[ ! -d "${RECSYS}/third_party/FBGEMM/fbgemm_gpu/experimental/hstu" ]]; then
  echo "ERROR: missing third_party/FBGEMM HSTU sources; re-run 02_clone_upstream.sh" >&2
  exit 1
fi
pushd "${RECSYS}/third_party/FBGEMM/fbgemm_gpu/experimental/hstu" >/dev/null
export HSTU_DISABLE_86OR89=FALSE
export HSTU_DISABLE_ARBITRARY=TRUE
export HSTU_DISABLE_LOCAL=TRUE
export HSTU_DISABLE_RAB=TRUE
export HSTU_DISABLE_DRAB=TRUE
export HSTU_DISABLE_FP16=TRUE
# amd64 / B200: enable sm120 build path used by upstream Dockerfile
export HSTU_DISABLE_120=TRUE
pip install --no-build-isolation --user .
popd >/dev/null

echo "==> [5/6] DynamicEmb"
pushd "${RECSYS}/corelib/dynamicemb" >/dev/null
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" python3 setup.py install --prefix="${DEPS}"
popd >/dev/null

echo "==> [6/6] examples/commons custom CUDA ops"
pushd "${RECSYS}/examples/commons" >/dev/null
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" python3 setup.py install --prefix="${DEPS}"
popd >/dev/null

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

echo "==> Install complete. PYTHONUSERBASE=${DEPS}"
INNER

chmod +x "${INSTALL_SCRIPT}"

# Persist user site under /raid/hstu/deps across runs
"${SCRIPT_DIR}/docker_run.sh" --privileged --name "${CONTAINER_NAME}-install" -- \
  bash -lc "
    export HSTU_CONTAINER_RAID=${CONTAINER_RAID}
    export TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}'
    export HSTU_ARCH_LIST='${HSTU_ARCH_LIST}'
    export PYTHONUSERBASE=${CONTAINER_RAID}/deps
    export PATH=${CONTAINER_RAID}/deps/bin:\$PATH
    export PYTHONPATH=${CONTAINER_RAID}/deps/lib/python3.12/site-packages:${CONTAINER_RAID}/deps/lib/python3.11/site-packages:${CONTAINER_RAID}/recsys-examples/examples:\$PYTHONPATH
    bash ${CONTAINER_RAID}/_install_train_inner.sh
  " 2>&1 | tee "${LOG}"

echo "==> Install log: ${LOG}"
echo "==> If IMPORT smoke printed INSTALL_OK, proceed to 04_preprocess.sh"
