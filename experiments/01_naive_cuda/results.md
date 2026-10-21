# Kernel 01 — Naive CUDA SGEMM Results

This file records the benchmark configuration and measured performance of the naive CUDA SGEMM kernel.

The goal of this kernel is to establish a clean GPU baseline before applying memory-coalescing, shared-memory and tiling optimizations.

---

## NVIDIA GeForce GTX 1650

### Configuration

| Parameter | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 |
| Architecture | Turing |
| Precision | FP32 |
| Block size | `32 × 32` = 1024 threads |
| Compiler | NVIDIA `nvcc` |
| Optimization | `-O3` |
| CUDA target | `sm_75` |
| Random seed | `42` |
| Timed runs | `5` |

Build command:

```cmd
nvcc -O3 -std=c++17 -arch=sm_75 naive_gemm.cu -o naive_gemm.exe
```

Kernel execution time is measured using CUDA events. Memory allocation and host-device transfers are excluded from the timing.

---

## Performance

| Matrix Size | Median Time | Median Performance | Verification |
|---:|---:|---:|:---:|
| `512 × 512` | `15.1828 ms` | **17.6802 GFLOP/s** | ✅ PASS |
| `1024 × 1024` | `89.3256 ms` | **24.0411 GFLOP/s** | ✅ PASS |
| `2048 × 2048` | `710.7577 ms` | **24.1712 GFLOP/s** | — |
| `4096 × 4096` | `5677.0332 ms` | **24.2096 GFLOP/s** | — |

Full-matrix correctness was verified against an FP64-accumulated CPU reference for `512 × 512` and `1024 × 1024`.

The larger matrices were used primarily to study steady-state kernel performance, without the expensive CPU reference calculation.

---

## CPU vs Naive CUDA

| Matrix Size | Naive CPU | Naive CUDA | Speedup |
|---:|---:|---:|---:|
| `512 × 512` | `1.2774 GFLOP/s` | `17.6802 GFLOP/s` | **13.8×** |
| `1024 × 1024` | `1.2882 GFLOP/s` | `24.0411 GFLOP/s` | **18.7×** |

Even this unoptimized CUDA kernel provides a substantial speedup over the naive CPU implementation.

However, once the matrix is large enough to fully occupy the GPU, performance settles at roughly:

> **24.2 GFLOP/s**

The near-identical throughput at `1024`, `2048`, and `4096` suggests that simply increasing the workload no longer improves utilization. The kernel has reached a steady-state bottleneck.

This is exactly what the next optimization will target: the global-memory access pattern of adjacent warp threads.

---

## Key Result

Best steady-state performance on the GTX 1650:

> **24.2096 GFLOP/s**

for `4096 × 4096`.

Compared with the GPU's theoretical FP32 capability, this still leaves a very large amount of performance unexplored.

That gap is where the optimization journey begins.

---

## RTX 5090 — Coming Soon . . .

The same kernels are also being benchmarked on an **NVIDIA GeForce RTX 5090**, and those results will soon be transferred from my GPU performance-engineering worklog into this repository.

This will allow the exact same SGEMM implementations to be compared across two very different NVIDIA GPU generations:

**Turing — GTX 1650**  
vs.  
**Blackwell — RTX 5090**

The interesting part will not just be the raw speedup. I also want to observe how the same kernel responds to architectural differences in compute throughput, memory systems, occupancy, and optimization strategy.

> **Same kernel. Same math. Turing vs Blackwell.**

Stay tuned.