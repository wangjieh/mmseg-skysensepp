#!/usr/bin/env python
"""Smoke-test and benchmark the modified SkySenseVisionTransformer backbone.

Put the modified skysense_vit.py back into your project, then run this script
from the project root.

Example:
    python test_skysense_backbone_flash.py \
        --module mmseg.models.backbones.skysense_vit \
        --img-size 512 --patch-size 16 --in-channels 10 \
        --embed-dims 768 --num-layers 12 --num-heads 12 \
        --batch-size 2 --force-flash

For a cheap smoke test, keep the defaults.
"""

import argparse
import importlib
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str,
                        default="mmseg.models.backbones.skysense_vit",
                        help="Python module path containing "
                             "SkySenseVisionTransformer.")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--in-channels", type=int, default=10)
    parser.add_argument("--embed-dims", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--force-flash", action="store_true",
                        help="Set force_flash_attn=True. Unsupported SDPA "
                             "shapes will fail instead of silently falling "
                             "back.")
    parser.add_argument("--compile", action="store_true",
                        help="Optionally run torch.compile on the backbone.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    mod = importlib.import_module(args.module)
    SkySenseVisionTransformer = getattr(mod, "SkySenseVisionTransformer")

    model = SkySenseVisionTransformer(
        img_size=(args.img_size, args.img_size),
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        embed_dims=args.embed_dims,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=4,
        out_indices=(args.num_layers - 1,),
        use_flash_attn=True,
        force_flash_attn=args.force_flash,
        with_cp=False,
        init_cfg=None,
    ).cuda().train()

    if args.compile:
        model = torch.compile(model, backend="inductor", mode="default")

    x = torch.randn(
        args.batch_size,
        args.in_channels,
        args.img_size,
        args.img_size,
        device="cuda")

    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("device:", torch.cuda.get_device_name())
    print("input:", tuple(x.shape))
    print("dtype:", dtype)
    print("force_flash_attn:", args.force_flash)

    # One forward/backward correctness smoke test.
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=dtype):
        outs = model(x)
        loss = sum(o.float().mean() for o in outs)
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    print("output shapes:", [tuple(o.shape) for o in outs])
    print("smoke loss:", float(loss.detach().cpu()))
    print("peak memory after smoke:",
          f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB")

    # Benchmark forward + backward because training speed is what matters.
    for _ in range(args.warmup):
        with torch.autocast(device_type="cuda", dtype=dtype):
            outs = model(x)
            loss = sum(o.float().mean() for o in outs)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.iters):
        with torch.autocast(device_type="cuda", dtype=dtype):
            outs = model(x)
            loss = sum(o.float().mean() for o in outs)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"train step: {elapsed / args.iters * 1000:.3f} ms/iter")
    print("peak memory:",
          f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
