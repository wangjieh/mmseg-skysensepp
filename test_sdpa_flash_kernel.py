#!/usr/bin/env python
"""Check which real ViT token lengths can run with PyTorch SDPA FlashAttention.

Run:
    python test_sdpa_flash_kernel.py

This does not import MMSeg. It only verifies the PyTorch/CUDA kernel for
several token lengths that are common in ViT segmentation.
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def bench_one(batch_size, num_heads, num_tokens, head_dim, dtype, iters, warmup):
    q = torch.randn(batch_size, num_heads, num_tokens, head_dim,
                    device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Force FLASH_ATTENTION so unsupported shapes fail immediately.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for _ in range(warmup):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iters):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    print(
        f"N={num_tokens:5d} | shape={tuple(out.shape)} | "
        f"{elapsed / iters * 1000:.3f} ms/iter | "
        f"peak_mem={torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[1024, 2304, 4096])
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("device:", torch.cuda.get_device_name())
    print("flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient_sdp_enabled:",
          torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp_enabled:", torch.backends.cuda.math_sdp_enabled())
    print("dtype:", dtype)

    for n in args.tokens:
        bench_one(args.batch_size, args.num_heads, n, args.head_dim,
                  dtype, args.iters, args.warmup)


if __name__ == "__main__":
    main()
