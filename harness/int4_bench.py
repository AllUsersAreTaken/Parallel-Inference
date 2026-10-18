"""On-card int4 (tinygemm/Marlin-class) vs fp16 matmul at small M, with L2 defeated.

Rotation: each measured iteration touches a *different* weight tensor, and the
per-shape working set is sized well above the 24 MB L2, so weights are read from
DRAM cold - matching a decode step, where the whole model is swept once and no
weight is reused across steps.

Weight bytes: N*K*0.5 for int4, N*K*2 for fp16.
"""
import math
import time
import torch

dev = "cuda"
torch.manual_seed(0)
TARGET_WORKING_BYTES = 256 << 20  # keep working set >> 24 MB L2


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def build_int4(N, K, group=128, inner=2):
    w_u8 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    packed = torch.ops.aten._convert_weight_to_int4pack(w_u8, inner)
    ng = K // group
    scales = torch.rand(ng, N, dtype=torch.bfloat16, device=dev)
    zeros = torch.zeros(ng, N, dtype=torch.bfloat16, device=dev)
    sz = torch.stack([scales, zeros], dim=-1).contiguous()  # [ng, N, 2]
    return packed, sz


SHAPES = [
    (3072, 3072, "q/o"),
    (1024, 3072, "k/v"),
    (8192, 3072, "down"),
    (3072, 8192, "gate/up"),
    (128256, 3072, "lm_head"),
]
MS = [1, 2, 4, 8, 16, 32]

print(f"device={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} L2=24MB")
for (N, K, tag) in SHAPES:
    wb_i4 = int(N * K * 0.5)
    wb_f16 = int(N * K * 2)
    reps_i4 = max(1, min(16, math.ceil(TARGET_WORKING_BYTES / wb_i4)))
    reps_f16 = max(1, min(16, math.ceil(TARGET_WORKING_BYTES / wb_f16)))

    i4 = [build_int4(N, K) for _ in range(reps_i4)]
    W16 = [torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(reps_f16)]
    print(f"# {tag} N={N} K={K} int4={wb_i4/1e6:.1f}MB x{reps_i4}  fp16={wb_f16/1e6:.1f}MB x{reps_f16}")

    for M in MS:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        ci = [0]
        def fn_i4():
            p, s = i4[ci[0] % reps_i4]; ci[0] += 1
            return torch._weight_int4pack_mm(x, p, 128, s)
        cf = [0]
        def fn_f16():
            w = W16[cf[0] % reps_f16]; cf[0] += 1
            return torch.nn.functional.linear(x, w)
        try:
            t_i4 = bench(fn_i4); ok = True
        except Exception as e:  # noqa: BLE001
            t_i4 = float("nan"); ok = False; err = str(e)[:50]
        t_f16 = bench(fn_f16)
        gi = wb_i4 / t_i4 / 1e9 if ok else float("nan")
        gf = wb_f16 / t_f16 / 1e9
        spd = (t_f16 / t_i4) if ok else float("nan")
        note = "" if ok else f"  [int4 FAILED: {err}]"
        print(f"{tag:9s} M={M:2d} | int4 {t_i4*1e6:8.1f} us {gi:6.1f} GB/s | "
              f"fp16 {t_f16*1e6:8.1f} us {gf:6.1f} GB/s | int4/fp16 {spd:4.2f}x{note}")
    del i4, W16
    torch.cuda.empty_cache()
