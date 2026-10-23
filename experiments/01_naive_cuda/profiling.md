# Kernel 01 — Nsight Compute Profiling

## Setup

| Parameter | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 |
| Matrix | `512 × 512` |
| Block | `32 × 32` |
| Kernel | `sgemm_naive` |
| Profiler | Nsight Compute 2025.4.1 |

---

## Key Metrics

| Metric | Value |
|---|---:|
| Compute Throughput | `5.86%` |
| Memory Throughput | `48.32%` |
| L1/TEX Throughput | `96.63%` |
| DRAM Throughput | `0.76%` |
| Achieved Occupancy | `98.28%` |
| Active Warps / Scheduler | `7.86` |
| Eligible Warps / Scheduler | `0.08` |
| No Eligible Warp | `97.87%` |

---

## What Nsight Shows

The kernel has **very high occupancy (~98%)**, so lack of resident warps is not the problem.

However, the schedulers almost never have a warp ready to issue:

```text
Active warps / scheduler   ≈ 7.86
Eligible warps / scheduler ≈ 0.08
```

Nsight reports that roughly **95.8% of the cycles between issued instructions are spent stalled on the local/global memory pipeline**.

This matches the access pattern of the naive kernel:

```text
A: strided across warp lanes     (Bad) 
B: broadcast-friendly            (Good)
```

So the kernel is not compute-bound. The load/store path is heavily stressed, leaving the CUDA cores underutilized.

---

## Bottleneck

```text
High occupancy
    ↓
Many resident warps
    ↓
Almost none eligible
    ↓
Memory pipeline stalls
    ↓
Low compute utilization
```

---

## Next Optimization

The next kernel changes the thread mapping so neighboring warp lanes access neighboring memory locations.

Goal:

> **Improve global-memory coalescing and reduce memory-pipeline stalls.**