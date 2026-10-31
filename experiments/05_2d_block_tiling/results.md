# Kernel 05 — 2D Block Tiling: Results

This document records the measured performance of our FP32 CUDA SGEMM implementation, from the initial 2D-tiled kernel through three focused autotuning sweeps.

Kernel design and reasoning are explained in [README.md](README.md). Nsight Compute findings will be documented separately in [profiling.md](profiling.md).

## 1. Benchmark Environment and Method

| Item | Configuration |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 (Turing, `sm_75`) |
| CUDA compiler | NVCC / CUDA 13.1 |
| Build flags | `-O3 -std=c++17 -arch=sm_75 -lineinfo` |
| Operation | FP32 SGEMM, `C = alpha * A * B + beta * C` |
| Benchmark inputs | Square matrices (`M=N=K`), `alpha=1`, `beta=0` |
| Timing | CUDA events; median of the stated timed runs |
| Throughput | `2 * N^3 / elapsed_seconds / 1e9` GFLOP/s |
| Correctness | Compared against a CPU reference with FP64 accumulation when `verify=1` |

Timings cover kernel execution, not host-device transfers, allocation, or CPU verification.

The initial four-size benchmark used **five timed runs** per size; the final 4096 autotuning comparisons used **nine timed runs**.

**Verification status:**
- `PASS`: Numerical verification was executed successfully.
- `SKIPPED`: Verification was disabled for that benchmark.

---

## 2. Initial 2D Block Tiling — 64×64×8

Our first 2D-tiled implementation used:

```cpp
BM = 64;
BN = 64;
BK = 8;

TM = 4;
TN = 4;

THREADS = 256;
```

Each thread computed a **4×4 output patch**, maintaining 16 accumulators in registers.

Compilation reported:
- 64 registers/thread.
- 4,352 bytes of shared memory/block.
- Zero register spill loads or stores.

### Initial benchmark results

| Matrix | Median time (ms) | Median GFLOP/s | Verification |
|---|---:|---:|---|
| 512×512 | 0.2869 | 935.6021 | PASS (0.000031) |
| 1024×1024 | 2.0152 | 1,065.6260 | PASS (0.000080) |
| 2048×2048 | 15.2106 | 1,129.4699 | SKIPPED |
| 4096×4096 | 96.7864 | **1,420.0230** | SKIPPED |

### Comparison against Kernel 04

| Matrix | Kernel 04 (GFLOP/s) | Initial Kernel 05 (GFLOP/s) | Speedup |
|---|---:|---:|---:|
| 512 | 526.3936 | 935.6021 | **1.78×** |
| 1024 | 643.2982 | 1,065.6260 | **1.66×** |
| 2048 | 676.3651 | 1,129.4699 | **1.67×** |
| 4096 | 878.5598 | 1,420.0230 | **1.62×** |

**Observation:** Even before autotuning, computing a 2D output patch per thread improved performance across all tested matrix sizes.

---

## 3. Autotuning Stage I — K-Tile Depth

Our first autotuning experiment fixed:

```cpp
BM = 64;
BN = 64;

TM = 4;
TN = 4;

THREADS = 256;
```

We varied only `BK`, testing 8, 16, and 32.

### Results

| BK | A tile | B tile | 512 GFLOP/s | 4096 time (ms) | 4096 GFLOP/s | Registers/thread* | SMEM/block |
|---|---|---|---:|---:|---:|---|---:|
| 8 | 64×8 | 8×64 | 934.1434 | 97.8698 | 1,404.3036 | 64 / 72 | 4,352 B |
| 16 | 64×16 | 16×64 | 1,032.0629 | 89.3106 | 1,538.8880 | 72 / 80 | 8,448 B |
| **32** | 64×32 | 32×64 | **1,065.6260** | **88.0519** | **1,560.8851** | 82 / 96 | 16,640 B |

All configurations passed correctness verification at 512×512 with a maximum absolute error of 0.000031.

No register spills were reported.

### Observations

**BK=8 → BK=16**

Performance at 4096 increased from 1,404.30 to 1,538.89 GFLOP/s.

This was consistent with our hypothesis that processing more K elements per tile could reduce the relative overhead of synchronization and loop control.

**BK=16 → BK=32**

Performance increased only slightly, reaching 1,560.89 GFLOP/s.

At the same time, register usage and shared-memory consumption increased.

**Conclusion:** BK=32 was the fastest configuration for the initial 64×64 output tile, but the improvement showed diminishing returns.

This motivated us to investigate the output tile shape rather than continue increasing BK.

---

## 4. Autotuning Stage II — Output Tile and Per-Thread Shape

Next, we explored different output tile dimensions and per-thread accumulator shapes.

We maintained **256 threads/block** and tested both rectangular and square output tiles.

### Tested configurations

| Output tile × BK | Thread tile | A tile | B tile | 512 GFLOP/s | 4096 time (ms) | 4096 GFLOP/s |
|---|---|---|---|---:|---:|---:|
| **128×64×16** | **8×4** | 128×16 | 16×64 | **1,101.4454** | **71.9987** | **1,908.9075** |
| 64×128×16 | 4×8 | 64×16 | 16×128 | 673.4592 | 110.6022 | 1,242.6417 |
| 128×128×8 | 8×8 | 128×8 | 8×128 | 728.9371 | 80.7427 | 1,702.1853 |
| 128×128×16 | 8×8 | 128×16 | 16×128 | 784.8622 | 76.2343 | 1,802.8500 |

All four configurations passed correctness verification at 512×512.

### Resource usage

| Configuration | Registers/thread* | Shared memory/block | Spills |
|---|---|---:|---|
| **8×4, BK=16** | **118 / 128** | 12,800 B | 0 |
| 4×8, BK=16 | 96 / 100 | 12,544 B | 0 |
| 8×8, BK=8 | 140 / 152 | 8,704 B | 0 |
| 8×8, BK=16 | 154 / 162 | 16,896 B | 0 |

### Important findings

**Finding 1 — The largest per-thread tile was not necessarily the fastest.**

An 8×8 tile maintains 64 output accumulators per thread, compared with 32 for 8×4.

Although this offers greater conceptual operand reuse, it also increases register requirements.

The 8×8 configurations did not outperform 8×4 in our experiments.

**Finding 2 — Rectangular and square output tiles behaved differently.**

The asymmetric 128×64 output tile outperformed both 128×128 configurations tested.

This demonstrated that simply increasing the output tile dimensions does not guarantee better performance.

**Finding 3 — 8×4 and 4×8 were not interchangeable in practice.**

Both configurations compute 32 outputs per thread and have equivalent conceptual operand reuse.

However, their measured performance differed substantially.

This suggests that thread mapping, memory-access efficiency, compiler instruction selection, register usage, or scheduling may affect performance.

The exact contributing factors require Nsight Compute investigation.

### Stage II winner

```cpp
BM = 128;
BN = 64;
BK = 16;

TM = 8;
TN = 4;

THREADS = 256;
```

**Performance: 1,908.9075 GFLOP/s at 4096×4096.**

This became our candidate for the final tuning stage.

---

## 5. Autotuning Stage III — Final BK Sweep for 8×4

Having identified the promising 128×64 output tile, we performed another controlled BK sweep.

Fixed parameters:

```cpp
BM = 128;
BN = 64;

TM = 8;
TN = 4;

THREADS = 256;
```

Tested `BK=8`, `BK=16`, and `BK=32`.

### Final autotuning results

| BK | A tile | B tile | 512 GFLOP/s | 4096 median time (ms) | 4096 median GFLOP/s |
|---|---|---|---:|---:|---:|
| 8 | 128×8 | 8×64 | 923.0422 | 78.7750 | 1,744.7032 |
| **16** | **128×16** | **16×64** | **1,112.2524** | **72.1797** | **1,904.1217** |
| 32 | 128×32 | 32×64 | 1,009.4594 | 77.2751 | 1,778.5662 |

The 4096 measurements used **nine-run median timing**.

All three configurations passed 512×512 numerical verification with a maximum absolute error of 0.000031.

### Register and shared-memory usage

| BK | Registers/thread* | Shared memory/block | Spill loads/stores |
|---|---|---:|---|
| 8 | 100 / 122 | 6,656 B | 0 |
| **16** | **118 / 128** | **12,800 B** | **0** |
| 32 | 129 / 153 | 25,088 B | 0 |

*Two register counts are shown because NVCC compiled two kernel specializations. These do not represent a measured range for one kernel launch.*

### Interpretation

**BK=8 → BK=16**

Increasing the K-tile depth improved 4096 performance by approximately 9.1%.

**BK=16 → BK=32**

Performance decreased by approximately 6.6%.

The BK=32 configuration consumed substantially more registers and shared memory.

For a 256-thread block, register allocation above 128 registers/thread prevents two such blocks from residing simultaneously on a GTX 1650 SM with 65,536 registers, even before considering allocation granularity.

Reduced occupancy is therefore a plausible contributor to the slowdown.

However, we will confirm the actual occupancy and stall behavior using Nsight Compute rather than treating this explanation as proven.

### Final selection

**BK=16 was the fastest of the three configurations tested for the 128×64, 8×4 layout.**

---

## 6. Reproducibility of the Winning Configuration

We measured our winning configuration in two separate tuning experiments.

| Experiment | Median time (ms) | Median GFLOP/s |
|---|---:|---:|
| Initial shape sweep | 71.9987 | **1,908.9075** |
| Final BK sweep | 72.1797 | **1,904.1217** |

The throughput measurements differ by approximately **0.25%**.

This supports the reproducibility of the observed improvement under our benchmark conditions.

For final comparison, we use the retested result of **1,904.1217 GFLOP/s**, while retaining 1,908.9075 GFLOP/s as the highest recorded median.

---

## 7. Complete SGEMM Optimization Progression

The following table summarizes performance across our CUDA GEMM optimization journey.

All values are **GFLOP/s**, measured on the GTX 1650.

| Matrix | K01 Naive | K02 Coalesced | K03 Shared Memory | K04 1D Tiling | K05 Initial 2D | K05 Tuned 2D |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 17.6802 | 258.5246 | 280.6305 | 526.3936 | 935.6021 | **1,112.2524** |
| 1024 | 24.0411 | 268.8581 | 314.3979 | 643.2982 | 1,065.6260 | — |
| 2048 | 24.1712 | 330.7028 | 362.4019 | 676.3651 | 1,129.4699 | — |
| 4096 | 24.2096 | 326.8628 | 364.5842 | 878.5598 | 1,420.0230 | **1,904.1217** |

A dash indicates that the corresponding tuned configuration has not yet been measured at that matrix size.

### Speedup at 4096×4096

| Comparison | Speedup |
|---|---:|
| Tuned K05 vs K01 Naive | **78.65×** |
| Tuned K05 vs K03 Shared Memory | **5.22×** |
| Tuned K05 vs K04 1D Tiling | **2.17×** |
| Tuned K05 vs Initial K05 | **1.34×** |

### Speedup at 512×512

Tuned Kernel 05 achieved **1,112.25 GFLOP/s**, approximately **2.11× faster** than Kernel 04.

---

## 8. Final Selected Kernel

After three autotuning stages, our selected configuration is:

```cpp
constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 16;

constexpr int TM = 8;
constexpr int TN = 4;

constexpr int THREADS = 256;
```

| Property | Value |
|---|---|
| A shared-memory tile | 128×16 |
| B shared-memory tile | 16×64 |
| C output tile | 128×64 |
| Outputs/thread | 8×4 = 32 |
| Threads/block | 256 (8 warps) |
| Static shared memory/block | 12,800 bytes |
| Accumulator values/thread | 32 FP32 |
| Register allocation | 118 / 128 registers/thread across two compiled variants |
| Compiler-reported spilling | None |

### Final performance

**GPU:** NVIDIA GeForce GTX 1650

**Workload:** 4096×4096 FP32 SGEMM

**Execution time:** 72.1797 ms

**Throughput:** 1,904.1217 GFLOP/s (1.904 TFLOP/s)

**Improvement over Kernel 04:** 2.17×

### Correctness status

- Tuned configuration: **PASS at 512×512**.
- Tuned configuration: Verification at 1024×1024 remains outstanding.
- Tuned 4096×4096 performance runs: Verification disabled.
- No compiler-reported register spills.

---

## 9. Reproducing the Winning Configuration

The original `block2d_gemm.cu` retains the initial 64×64×8 baseline.

Our selected tunable implementation is `block2d_shapes.cu`.

Compile using the Visual Studio 2022 x64 Native Tools Command Prompt:

```cmd
nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Xptxas=-v -DBM_SIZE=128 -DBN_SIZE=64 -DBK_SIZE=16 -DTM_SIZE=8 -DTN_SIZE=4 block2d_shapes.cu -o block2d_best.exe
```

Run verification:

```cmd
block2d_best.exe 512 5 1
block2d_best.exe 1024 5 1
```

Benchmark:

```cmd
block2d_best.exe 4096 9 0
```

The 1024 verification command is included as an outstanding test, not as a previously completed measurement.

---

## 10. Conclusion and Next Steps

Kernel 05 demonstrates that **2D register tiling combined with systematic tile-shape and K-depth tuning can substantially improve SGEMM performance**.

Our experiments revealed several important optimization tradeoffs:

- More per-thread arithmetic can increase operand reuse, but also increases register requirements.
- Larger K-tiles can reduce the relative cost of synchronization, but may consume enough resources to hurt parallelism.
- Rectangular output tiles can outperform square tiles.
- Mathematically equivalent per-thread output shapes may generate very different GPU execution behavior.
- The largest tile or highest theoretical reuse does not necessarily produce the fastest kernel.

Our selected **128×64×16 output/reduction tile with 8×4 outputs/thread** achieved approximately **1.90 TFLOP/s**, exceeding Kernel 04 by **2.17×**.

The next stage is to profile the selected kernel.

- Achieved occupancy and register pressure.
- Shared-memory load/store instruction counts.
- MIO Throttle.
- Long Scoreboard.
- Scheduler eligibility and issue efficiency.