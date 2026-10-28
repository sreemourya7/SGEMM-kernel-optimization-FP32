# Kernel 04 - 1D Block Tiling: Results

## Benchmark setup

| Item | Value |
|---|---|
| GPU | NVIDIA GTX 1650, Turing (`sm_75`) |
| Precision | FP32 SGEMM, `C = alpha * A * B + beta * C` |
| Benchmark values | `M=N=K`, `alpha=1`, `beta=0` |
| Compiler | NVCC / CUDA 13.1, `-O3 -std=c++17 -arch=sm_75 -lineinfo` |
| Kernel 04 tile | `BM=64, BN=64, BK=8, TM=8` |
| Block | 512 threads, 8 output elements/thread |
| Timing | CUDA events, median of five timed runs |
| Reported work | `2*N^3` floating-point operations |

Run from the Kernel 04 experiment directory:

```cmd
nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo block1d_gemm.cu -o block1d_gemm.exe
block1d_gemm.exe 512 5 1
block1d_gemm.exe 1024 5 1
block1d_gemm.exe 2048 5 0
block1d_gemm.exe 4096 5 0
```

`verify=1` checks against a CPU FP64-accumulation reference. Verification **passed** for `512` and `1024`; larger sizes were run with verification **skipped** (not independently validated in this benchmark).

## 1. Kernel 04 measured results

| Matrix | Median time | Median throughput | Verification |
|---|---:|---:|---|
| `512 x 512` | **0.5100 ms** | **526.3936 GFLOP/s** | PASS |
| `1024 x 1024` | **3.3382 ms** | **643.2982 GFLOP/s** | PASS |
| `2048 x 2048` | **25.4003 ms** | **676.3651 GFLOP/s** | SKIPPED |
| `4096 x 4096` | **156.4366 ms** | **878.5598 GFLOP/s** | SKIPPED |

**Best measured throughput:** `878.56 GFLOP/s` at `4096 x 4096`.

## 2. Full optimization progression

All values are GFLOP/s. Previous-kernel results are taken from the existing [Kernel 02](../02_global_memory_coalescing/results.md) and [Kernel 03](../03_shared_memory_tiling/results.md) benchmark logs.

| Matrix | CPU baseline | K01 Naive | K02 Coalesced | K03 SMEM tiled | **K04 1D blocked** |
|---|---:|---:|---:|---:|---:|
| 512 | 1.2774 | 17.6802 | 258.5246 | 280.6305 | **526.3936** |
| 1024 | 1.2882 | 24.0411 | 268.8581 | 314.3979 | **643.2982** |
| 2048 | 0.1393 | 24.1712 | 330.7028 | 362.4019 | **676.3651** |
| 4096 | - | 24.2096 | 326.8628 | 364.5842 | **878.5598** |

The CPU result is from a single-threaded reference; there is no recorded CPU benchmark at `4096`. The CUDA kernels use different block and tile mappings, so this table compares the actual measured implementations, not an isolated single parameter.

## 3. Speedup from Kernel 03 to Kernel 04

| Matrix | K03 time | K04 time | K03 throughput | K04 throughput | **K04/K03** |
|---|---:|---:|---:|---:|---:|
| 512 | 0.9565 ms | **0.5100 ms** | 280.63 | **526.39** | **1.88x** |
| 1024 | 6.8305 ms | **3.3382 ms** | 314.40 | **643.30** | **2.05x** |
| 2048 | 47.4056 ms | **25.4003 ms** | 362.40 | **676.37** | **1.87x** |
| 4096 | 376.9745 ms | **156.4366 ms** | 364.58 | **878.56** | **2.41x** |

At `4096`, Kernel 04 is approximately **36.3x faster than Kernel 01** and **2.69x faster than Kernel 02**, measured by achieved GFLOP/s on this machine.

## 4. Did the 1D tiling optimization work?

**Yes.** Kernel 04 was motivated by a specific profiler finding: Kernel 03's compute loop created high shared-memory instruction demand and dominant MIO Throttle. Assigning eight C values to a thread was intended to reuse one B operand for eight FMAs and reduce the number of shared-memory load instructions per unit of arithmetic.

The following Nsight results are measured at `512 x 512`:

| Evidence | Kernel 03 | Kernel 04 | Change |
|---|---:|---:|---:|
| Shared-memory load instructions | 5,242,880 | **1,572,864** | **-70.0%** |
| MIO Throttle (warp cycles per issued instruction) | 23.60 | **4.62** | **-80.4%** |
| Warp cycles per issued instruction | 38.44 | **16.23** | **-57.8%** |
| Registers/thread | 44 | **66** | Increased |
| Achieved occupancy | 99.78% | **50.20%** | Decreased |
| Standalone throughput | 280.63 | **526.39 GFLOP/s** | **1.88x** |

The larger `4096 x 4096` profile shows a similar MIO improvement: **23.80 -> 4.03** (-83.1%) alongside a **2.41x** speedup in standalone throughput.

**Interpretation:** The kernel performs more useful FP32 work per shared-memory load instruction and substantially reduces normalized MIO stall pressure. This outweighs the lower occupancy on our tested workloads. Because tile and block dimensions also changed, the data supports the effectiveness of Kernel 04 as an implementation rather than proving a perfectly controlled register-reuse-only speedup.

## 5. Remaining work

At `4096`, the two largest observed warp stalls are **Long Scoreboard = 4.09** and **MIO Throttle = 4.03** cycles per issued instruction. Source-level profiling at `512` associates Long Scoreboard with a B tile global-load-to-shared-store dependency and MIO stalls with shared-memory loads in the compute loop.

Kernel 05 will investigate **2D register tiling** (`4 x 4` outputs/thread) to reuse both A and B within a thread, potentially lowering shared-memory instruction demand further. Whether that also reduces Long Scoreboard stalls must be verified experimentally.

For the complete investigation, see [profiling.md](profiling.md). The [raw Nsight Compute measurements](https://github.com/sreemourya7/cuda-nsight-compute-profiling/blob/main/kernels/04_1d_tiling_cuda/README.md) are maintained in the separate profiling repository.
