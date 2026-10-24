# Kernel 02 — Global Memory Coalescing: Results

## 1. Benchmark Setup

| Parameter | Value |
|---|---|
| GPU | NVIDIA GTX 1650 (Turing, SM 7.5) |
| Compiler | NVCC, CUDA 13.1 |
| Precision | FP32 |
| Block Size | 1024 threads (32 × 32 tile) |
| Timing | CUDA Events, median of 5 runs |
| Optimization | `-O3 -arch=sm_75` |

## 2. Compile & Execute

Open **x64 Native Tools Command Prompt for VS 2022**.

```cmd
cd /d "%USERPROFILE%\OneDrive\Desktop\SGEMM-kernel-optimization-FP32\experiments\02_global_memory_coalescing"
```

**Compile:**

```cmd
nvcc -O3 -std=c++17 -arch=sm_75 gmem_gemm.cu -o gmem_gemm.exe
```

**Run:**

```cmd
gmem_gemm.exe 512 5 1
gmem_gemm.exe 1024 5 1
gmem_gemm.exe 2048 5 0
gmem_gemm.exe 4096 5 0
```

Arguments: `matrix_size timed_runs verify(0/1)`

## 3. Kernel 02 — Benchmark Results

| Matrix | Median Time (ms) | GFLOP/s | Verification |
|---|---:|---:|---|
| 512 | 1.0383 | 258.52 | PASS |
| 1024 | 7.9874 | 268.86 | PASS |
| 2048 | 51.9496 | 330.70 | Skipped |
| 4096 | 420.4790 | 326.86 | Skipped |

Correctness was verified against the CPU reference for 512 and 1024. Larger sizes were benchmarked without CPU verification.

## 4. Cumulative Performance Comparison

All performance values are in **GFLOP/s**.

| Matrix | CPU Baseline | Kernel 01: Naive | Kernel 02: Coalesced |
|---|---:|---:|---:|
| 512 | 1.2774 | 17.6802 | **258.5246** |
| 1024 | 1.2882 | 24.0411 | **268.8581** |
| 1536 | 0.6443 | — | — |
| 2000 | 0.4161 | — | — |
| 2048 | 0.1393 | 24.1712 | **330.7028** |
| 4096 | — | 24.2096 | **326.8628** |

*CPU baseline is single-threaded. Dashes indicate sizes not benchmarked.*

### Speedup Over Previous CUDA Kernel

| Matrix | Kernel 01 | Kernel 02 | Speedup |
|---|---:|---:|---:|
| 512 | 17.68 | 258.52 | **14.62×** |
| 1024 | 24.04 | 268.86 | **11.18×** |
| 2048 | 24.17 | 330.70 | **13.68×** |
| 4096 | 24.21 | 326.86 | **13.50×** |

## 5. Observation

Changing the thread mapping to favor coalesced global-memory accesses improved performance by **11–15×** compared with the naive kernel.

The best measured throughput was **330.70 GFLOP/s** at matrix size 2048.

The next step is to profile Kernel 02 using Nsight Compute and investigate how memory transactions, LG Throttle, warp eligibility, and L1/TEX utilization changed.

## RTX 5090 — Coming Soon . . .

The same kernels are also being benchmarked on an **NVIDIA GeForce RTX 5090**, and those results will soon be transferred from my GPU performance-engineering worklog into this repository.

This will allow the exact same SGEMM implementations to be compared across two very different NVIDIA GPU generations:

**Turing — GTX 1650**  
vs.  
**Blackwell — RTX 5090**

The interesting part will not just be the raw speedup. I also want to observe how the same kernel responds to architectural differences in compute throughput, memory systems, occupancy, and optimization strategy.

> **Same kernel. Same math. Turing vs Blackwell.**

Stay tuned.
