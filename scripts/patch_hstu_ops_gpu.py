#!/usr/bin/env python3
"""Patch installed hstu/hstu_ops_gpu.py to skip register_fake for missing ops.

With HSTU_ARCH_LIST containing 8.0 but not 9.0, the CUDA .so only defines
fbgemm::hstu_varlen_fwd_80. Importing hstu.hstu_ops_gpu then crashes on:

  RuntimeError: operator fbgemm::hstu_varlen_fwd_90 does not exist

On B200 the real compute path is hstu.hstu_blackwell; these fakes are only
needed for torch.export. Skipping missing ops is safe for training.
"""
from __future__ import annotations

import importlib.util
import os
import sys


PATCH_MARKER = "# hstu_opt: safe_register_fake"

SAFE_HEADER = '''# Fake (abstract) implementations for fbgemm::hstu_varlen_fwd_* ops.
# These are required by torch.export / FakeTensor tracing.
# Registered via set_python_module("hstu.hstu_ops_gpu") in the C++ TORCH_LIBRARY_FRAGMENT.
# hstu_opt: safe_register_fake

import torch


def _safe_register_fake(qualname):
    """Like torch.library.register_fake, but no-op if the op was not compiled in."""

    def decorator(fn):
        try:
            return torch.library.register_fake(qualname)(fn)
        except RuntimeError as exc:
            if "does not exist" in str(exc):
                return fn
            raise

    return decorator


@_safe_register_fake("fbgemm::hstu_varlen_fwd_80")
def _hstu_varlen_fwd_80_fake(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    seqused_q, seqused_k,
    max_seqlen_q, max_seqlen_k,
    scaling_seqlen=-1,
    num_contexts=None, num_targets=None,
    target_group_size=1,
    window_size_left=-1, window_size_right=-1,
    alpha=1.0,
    rab=None, func=None,
    kv_cache=None, page_offsets=None, page_ids=None, last_page_lens=None,
):
    return torch.empty_like(v), torch.empty(0, device=q.device, dtype=torch.float32)


@_safe_register_fake("fbgemm::hstu_varlen_fwd_90")
def _hstu_varlen_fwd_90_fake(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    seqused_q, seqused_k,
    max_seqlen_q, max_seqlen_k,
    scaling_seqlen=-1,
    num_contexts=None, num_targets=None,
    target_group_size=1,
    window_size_left=-1, window_size_right=-1,
    alpha=1.0,
    rab=None, func=None,
    quant_mode=-1, output_dtype=-1,
    vt=None, cu_seqlens_vt_descale=None,
    q_descale=None, k_descale=None, v_descale=None, vt_descale=None,
    cu_seqlens_q_block_descale=None, cu_seqlens_kv_block_descale=None,
):
    if output_dtype == 0:
        out_dtype = torch.bfloat16
    elif output_dtype == 1:
        out_dtype = torch.float16
    else:
        out_dtype = v.dtype
    return v.new_empty(v.shape, dtype=out_dtype), torch.empty(0, device=q.device, dtype=torch.float32)
'''


def find_hstu_ops_gpu() -> str:
    spec = importlib.util.find_spec("hstu")
    if spec is None or not spec.origin:
        raise SystemExit("hstu package not found on sys.path")
    path = os.path.join(os.path.dirname(spec.origin), "hstu_ops_gpu.py")
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}")
    if "examples/hstu" in path:
        raise SystemExit(f"refusing to patch examples tree: {path}")
    return path


def main() -> int:
    path = find_hstu_ops_gpu()
    with open(path, "r", encoding="utf-8") as f:
        old = f.read()
    if PATCH_MARKER in old:
        print(f"already patched: {path}")
        return 0
    bak = path + ".bak"
    if not os.path.exists(bak):
        with open(bak, "w", encoding="utf-8") as f:
            f.write(old)
        print(f"backup: {bak}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SAFE_HEADER)
    print(f"patched: {path}")
    # Verify import
    import importlib

    if "hstu.hstu_ops_gpu" in sys.modules:
        del sys.modules["hstu.hstu_ops_gpu"]
    importlib.invalidate_caches()
    import hstu.hstu_ops_gpu  # noqa: F401

    print("import hstu.hstu_ops_gpu: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
