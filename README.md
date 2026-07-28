# hstu_opt — HSTU 8×B200 training + nsys/ncu profiling

Scripts to clone [NVIDIA/recsys-examples](https://github.com/NVIDIA/recsys-examples) HSTU training onto an 8×B200 node, prepare **MovieLens-20M**, run **ranking** and **retrieval** on 8 GPUs, and profile with host **nsys** / **ncu**.

Designed for a host where you cannot SSH from your laptop interactively: push this repo, pull on the box, run scripts, paste text logs back.

## Target host assumptions

- 8× NVIDIA B200, driver/CUDA compatible with NGC PyTorch `26.05-py3`
- x86_64 Linux (e.g. `b200-50`)
- Docker installed and able to pull `nvcr.io` (NGC login if required)
- Host `nsys` and `ncu` already on `PATH`
- Large disk at `/raid` (default workspace root `/raid/hstu`)

## Layout

```
configs/          # gin configs (smoke + profile windows)
scripts/
  00_env.sh                 # shared paths / image / NPROC
  docker_run.sh             # common docker run helper
  01_pull_image.sh
  02_clone_upstream.sh
  03_install_train.sh       # training-only deps (no Triton)
  04_preprocess.sh          # ml-20m
  05_train_ranking.sh
  06_train_retrieval.sh
  07_nsys_ranking.sh
  08_nsys_retrieval.sh
  09_ncu_ranking.sh         # single GPU
  10_ncu_retrieval.sh       # single GPU
  11_collect_artifacts.sh
```

Upstream `recsys-examples` is **cloned on the box** under `/raid/hstu/recsys-examples` (not vendored here).

## One-time setup (on the box)

```bash
# place this repo somewhere convenient
sudo mkdir -p /raid/hstu && sudo chown "$USER":"$USER" /raid/hstu
cd /raid/hstu
git clone <your-hstu_opt-remote> hstu_opt
cd hstu_opt

chmod +x scripts/*.sh
source scripts/00_env.sh
print_env

# NGC registry (if the pull is unauthorized):
#   docker login nvcr.io
#   # username: $oauthtoken   password: <NGC API key>

# Either step-by-step:
./scripts/01_pull_image.sh      # needs NGC/docker access
./scripts/02_clone_upstream.sh  # recursive clone + submodules
./scripts/03_install_train.sh   # long: FBGEMM / TorchRec / HSTU / DynamicEmb
./scripts/04_preprocess.sh      # downloads MovieLens-20M

# Or all four:
# ./scripts/run_setup.sh
```

Optional overrides before sourcing `00_env.sh`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `HSTU_RAID_ROOT` | `/raid/hstu` | All data / deps / profiles |
| `NGC_IMAGE` | `nvcr.io/nvidia/pytorch:26.05-py3` | Training container |
| `NPROC` | `8` | `torchrun --nproc_per_node` |
| `RECSYS_REF` | `main` | Upstream git ref |

## Smoke train (validate install)

```bash
./scripts/05_train_ranking.sh smoke
./scripts/06_train_retrieval.sh smoke
```

If either fails, paste the log path printed at the end (under `/raid/hstu/logs/`) or the last ~80 lines.

## Profiling

Profile gins use `TrainerArgs.profile=True` with a short `cudaProfilerApi` window (steps 20–30, max 40 iters).

```bash
# 8-GPU system traces
./scripts/07_nsys_ranking.sh
./scripts/08_nsys_retrieval.sh

# Single-GPU kernel deep dive (GPU 0)
./scripts/09_ncu_ranking.sh
./scripts/10_ncu_retrieval.sh

# Optional: heavier ncu metric set
NCU_SET=full ./scripts/09_ncu_ranking.sh

# Optional: filter kernels after you know names from nsys
NCU_KERNEL_REGEX='regex:.*hstu.*' ./scripts/09_ncu_ranking.sh
```

Profilers run **on the host** and wrap a **privileged** container so CUPTI can capture GPU activity.

Outputs:

- `/raid/hstu/profiles/nsys/*.nsys-rep`
- `/raid/hstu/profiles/ncu/*.ncu-rep`
- `/raid/hstu/logs/*.log`

## Collect paste-back summary

```bash
./scripts/11_collect_artifacts.sh
```

This prints `PASTE_ME.txt` (recent log tails) and writes a tarball under `/raid/hstu/profiles/summaries/`. Copy/paste `PASTE_ME.txt` (and `environment.txt` if useful) into chat. Leave binary `.nsys-rep` / `.ncu-rep` on `/raid`.

## Gin configs

| File | Use |
|------|-----|
| `configs/movielen_ranking_smoke.gin` | Ranking smoke, 50 iters, no profiler |
| `configs/movielen_retrieval_smoke.gin` | Retrieval smoke |
| `configs/movielen_ranking_profile.gin` | Ranking nsys/ncu window |
| `configs/movielen_retrieval_profile.gin` | Retrieval nsys/ncu window |

Dataset path is set to `/raid/hstu/data` (container mount of the raid tree).

## Notes

- Install script skips Triton / FlexKV / NVE / AOTI (training-only).
- First `03_install_train.sh` can take a long time (FBGEMM CUDA build).
- `fbgemm_gpu_hstu` build defaults for this B200 kit:
  - `MAX_JOBS=4` (use `2` if OOM)
  - `HSTU_ARCH_LIST=10.0` (Blackwell only; skips Hopper sm90)
  - `HSTU_DISABLE_FP8=TRUE` (MovieLens bf16 does not need e4m3 kernels)
  ```bash
  MAX_JOBS=2 ./scripts/03_install_train.sh
  ```
  Re-runs skip packages that already import cleanly.
- If DynamicEmb fails with `No module named 'fbgemm_gpu'` while TorchRec is installed: an older `setup.py --prefix` put FBGEMM under `deps/local/lib/python3.12/dist-packages`. Current scripts put that path on `PYTHONPATH` and prefer `pip --user` wheels. Just re-run:
  ```bash
  ./scripts/03_install_train.sh
  ```
  Force-rebuild HSTU only if needed:
  ```bash
  rm -rf /raid/hstu/deps/lib/python3.12/site-packages/hstu* \
         /raid/hstu/deps/lib/python3.12/site-packages/fbgemm_gpu_hstu* \
         /raid/hstu/recsys-examples/third_party/FBGEMM/fbgemm_gpu/experimental/hstu/build
  MAX_JOBS=2 ./scripts/03_install_train.sh
  ```
- If nsys reports “No reports were generated”, confirm `TrainerArgs.profile=True` and that the container was started with `--privileged` (already set in 07–10).
- `MASTER_PORT` defaults to `6000`; change if something else is bound.
