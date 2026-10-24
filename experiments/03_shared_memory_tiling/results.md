# Kernel 03 — Shared Memory Tiling: Results

## Benchmark Setup

**GPU:** GTX 1650 | **CUDA:** 13.1 | **Precision:** FP32

**Compilation:**

```cmd
nvcc -O3 -std=c++17 -arch=sm_75 smem_gemm.cu -o smem_gemm.exe
```

**Execution:**

```cmd
smem_gemm.exe 512 5 1
smem_gemm.exe 1024 5 1
smem_gemm.exe 2048 5 0
smem_gemm.exe 4096 5 0
```

CUDA Event timing; median of five runs.

## Kernel 03 Results

| Matrix | Median Time | GFLOP/s | Verification |
|---|---:|---:|---|
| 512 | 0.9565 ms | 280.63 | PASS |
| 1024 | 6.8305 ms | 314.40 | PASS |
| 2048 | 47.4056 ms | 362.40 | Skipped |
| 4096 | 376.9745 ms | 364.58 | Skipped |

## Cumulative Performance Comparison

All values in GFLOP/s.

| Matrix | CPU | K01 Naive | K02 Coalesced | K03 SMEM |
|---|---:|---:|---:|---:|
| 512 | 1.2774 | 17.68 | 258.52 | **280.63** |
| 1024 | 1.2882 | 24.04 | 268.86 | **314.40** |
| 1536 | 0.6443 | — | — | — |
| 2000 | 0.4161 | — | — | — |
| 2048 | 0.1393 | 24.17 | 330.70 | **362.40** |
| 4096 | — | 24.21 | 326.86 | **364.58** |

*CPU is a single-threaded naive implementation.*

## Speedup Over Kernel 02

| Matrix | Speedup |
|---|---:|
| 512 | 1.09× |
| 1024 | 1.17× |
| 2048 | 1.10× |
| 4096 | 1.12× |

Shared-memory tiling improved throughput across all measured sizes, but the gains were modest compared with global-memory coalescing.

**Best result:** 364.58 GFLOP/s at 4096 × 4096.

## RTX 5090 — Coming Soon . . .

The same kernels are also being benchmarked on an **NVIDIA GeForce RTX 5090**, and those results will soon be transferred from my GPU performance-engineering worklog into this repository.

This will allow the exact same SGEMM implementations to be compared across two very different NVIDIA GPU generations:

**Turing — GTX 1650**  
vs.  
**Blackwell — RTX 5090**

The interesting part will not just be the raw speedup. I also want to observe how the same kernel responds to architectural differences in compute throughput, memory systems, occupancy, and optimization strategy.

> **Same kernel. Same math. Turing vs Blackwell.**

Stay tuned.
