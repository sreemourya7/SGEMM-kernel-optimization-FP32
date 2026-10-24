# Kernel 02 — Nsight Compute Profiling

[View Raw Nsight Compute Metrics](https://github.com/sreemourya7/cuda-nsight-compute-profiling/blob/main/kernels/02_gmem_cuda/README.md)

## What Changed?

Kernel 02 improved performance by **~14.6×** at `512 × 512`, simply by changing the thread-to-output mapping to achieve coalesced global-memory accesses.

But did this actually reduce our previous bottleneck?

## What Nsight Shows

| Metric | Kernel 01 | Kernel 02 |
|---|---:|---:|
| SM Throughput | 5.86% | **86.47%** |
| L1/TEX Throughput | 96.63% | 89.58% |
| Achieved Occupancy | 98.28% | 98.52% |
| Eligible Warps / Scheduler | 0.08 | **1.13** |
| No Eligible Warp | 97.87% | **68.54%** |
| LG Throttle | 353.8 cycles | **18.9 cycles** |
| Mem Pipes Busy | 5.84% | **86.02%** |

Occupancy remained almost identical, but eligible warps increased significantly. More importantly, the average LG Throttle stall contribution dropped from **353.8 to 18.9 cycles per issued instruction**.

Coalesced loads require fewer memory sectors, potentially reducing L1/TEX processing work. This is consistent with the significant improvement in scheduler activity and compute throughput.

## What's Still Wrong?

Despite the improvement, schedulers still have **no eligible warp on 68.54% of cycles**.

LG Throttle remains the dominant stall reason, accounting for **75.3% of the average warp issue interval**.

Interestingly, Mem Pipes Busy increased to **86.02%**, suggesting that memory-instruction issue throughput is now heavily utilized.

Our loads are more efficient, but we're still issuing global-memory instructions repeatedly inside the GEMM loop, even when the data is already cached.

## Next Optimization — Shared Memory Tiling

**Remember: Coalescing makes individual global loads more efficient. Tiling with shared memory reduces how often we need those global loads.**

```text
Kernel 02
Coalesced global loads
        ↓
Repeated loads inside GEMM loop
        ↓
High memory-instruction pressure
        ↓
LG Throttle remains


Kernel 03 — Hypothesis
Cooperatively load A and B tiles
        ↓
Store tiles in shared memory
        ↓
Reuse data across threads
        ↓
Fewer redundant global loads
        ↓
Potentially lower LG Throttle
```

**Next:** shared-memory tiling.