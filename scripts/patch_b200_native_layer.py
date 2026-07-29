#!/usr/bin/env python3
"""Patch recsys-examples create_hstu_config to honor HSTU_FORCE_NATIVE.

Blackwell fused_hstu_op raises:
  ValueError: Blackwell fused_hstu_op does not support contextual tokens

MovieLens ranking/retrieval use contextual features, and TP=1 selects FUSED.
This patch forces HSTULayerType.NATIVE when HSTU_FORCE_NATIVE=1 (default).
"""
from __future__ import annotations

import os
import sys

MARKER = "# hstu_opt: HSTU_FORCE_NATIVE"
DEFAULT_UTILS = (
    "/raid/hstu/recsys-examples/examples/hstu/training/trainer/utils.py"
)


def patch_file(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        print(f"already patched: {path}")
        return

    needle = """    layer_type = None
    if tensor_model_parallel_args.tensor_model_parallel_size == 1:
        layer_type = HSTULayerType.FUSED
    else:
        layer_type = HSTULayerType.NATIVE
"""
    if needle not in src:
        raise SystemExit(
            f"Could not find layer_type selection block in {path}. "
            "Upstream may have changed; update patch_b200_native_layer.py."
        )

    replacement = """    layer_type = None
    if tensor_model_parallel_args.tensor_model_parallel_size == 1:
        layer_type = HSTULayerType.FUSED
    else:
        layer_type = HSTULayerType.NATIVE
    # hstu_opt: HSTU_FORCE_NATIVE
    # Blackwell fused_hstu_op does not support contextual tokens (MovieLens).
    import os as _hstu_opt_os
    if _hstu_opt_os.environ.get("HSTU_FORCE_NATIVE", "1") == "1":
        layer_type = HSTULayerType.NATIVE
"""
    bak = path + ".bak_hstu_opt"
    if not os.path.exists(bak):
        with open(bak, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"backup: {bak}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(needle, replacement, 1))
    print(f"patched: {path}")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "HSTU_TRAINER_UTILS", DEFAULT_UTILS
    )
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}")
    patch_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
