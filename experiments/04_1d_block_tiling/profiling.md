# Kernel 04 - Nsight Compute Profiling: From 1D Reuse to 2D Register Tiling

**GPU:** NVIDIA GTX 1650 (Turing, `sm_75`)  
**Kernel:** `sgemm_block1d` - `BM=64, BN=64, BK=8, TM=8`  
**Tools:** CUDA 13.1, Nsight Compute CLI, NVCC `-O3 -lineinfo`  
**Raw data:** [Kernel 04 profiler commands and metrics](https://github.com/sreemourya7/cuda-nsight-compute-profiling/blob/main/kernels/04_1d_tiling_cuda/README.md)

## 1. Our Starting Point: Kernel 03 Was MIO-Throttled

Shared-memory tiling in Kernel 03 made A and B available for reuse within a block, but **one thread still accumulated only one output element**. Repeated shared-memory operand loads created heavy instruction demand.

At `512 x 512`, Kernel 03 had:

| Metric | Kernel 03 |
|---|---:|
| Achieved occupancy | 99.78% |
| Active warps/scheduler | 7.93 |
| No eligible warp cycles | 79.37% |
| Warp cycles per issued instruction | 38.44 |
| MIO Throttle, cycles per issued instruction | **23.60** |
| Shared-memory load instructions | **5,242,880** |
| Standalone throughput | **280.63 GFLOP/s** |

SourceCounters linked many of those MIO samples to shared-memory `LDS` instructions in Kernel 03's GEMM inner loop. The hypothesis was **not** to increase occupancy: it was already near maximum. Instead, we wanted to perform more FMAs for each shared-memory operand read.

**Kernel 04's experiment:** Give a thread eight vertically adjacent C outputs and reuse one B value across eight FMAs, keeping eight C accumulators in registers.

## 2. First Check: Did 1D Tiling Make the Kernel Faster?

Standalone CUDA-event measurements (median of five timed runs):

| Matrix | Kernel 03 | Kernel 04 | K04 / K03 |
|---|---:|---:|---:|
| 512 | 280.63 GFLOP/s | **526.39 GFLOP/s** | **1.88x** |
| 1024 | 314.40 GFLOP/s | **643.30 GFLOP/s** | **2.05x** |
| 2048 | 362.40 GFLOP/s | **676.37 GFLOP/s** | **1.87x** |
| 4096 | 364.58 GFLOP/s | **878.56 GFLOP/s** | **2.41x** |

The performance gain is real in the measured workload, but a benchmark alone does not tell us *why* it happened. We profiled `512 x 512` and `4096 x 4096`.

## 3. Occupancy Fell - Yet Performance Increased

We began with the basic Nsight Compute collection:

```cmd
ncu --kernel-name sgemm_block1d --launch-count 1 block1d_gemm.exe 512 1 0
ncu --kernel-name sgemm_block1d --launch-count 1 block1d_gemm.exe 4096 1 0
```

| Metric | K03 at 512 | K04 at 512 | K04 at 4096 |
|---|---:|---:|---:|
| Registers/thread | 44 | **66** | **66** |
| Achieved occupancy | 99.78% | **50.20%** | **49.99%** |
| Active warps/SM | 31.93 | **16.06** | **16.00** |
| Static shared memory/block | ~8.19 KB | **4.10 KB** | **4.10 KB** |
| SM Throughput (Nsight SOL) | 69.13% | **42.73%** | **49.02%** |
| DRAM Throughput (Nsight SOL) | 10.71% | **12.31%** | **13.37%** |

With 512 threads/block and 66 registers/thread, the Turing SM is **register-limited to one resident block**, or 16 active warps. Neither local nor shared-memory spilling was reported.

**Interpretation:** More per-thread computation increased register demand and halved occupancy, yet the standalone GFLOP/s increased substantially. Occupancy measures resident warps; it does not tell us how efficiently each warp performs useful work. Similarly, Nsight's **SM Throughput (%) is not the achieved FP32 GFLOP/s divided by theoretical FP32 peak**.

## 4. Scheduler and Warp Stalls: Did MIO Throttle Improve?

We collected scheduler, warp state, and memory workload sections with detailed warp-state breakdowns:

```cmd
ncu --kernel-name sgemm_block1d --launch-count 1 --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --print-details all block1d_gemm.exe 4096 1 0
```

The same sections were collected at 512, and `--section WarpStateStats --print-details all` revealed the per-stall breakdown.

### 4.1 The 512 x 512 comparison

| Metric | Kernel 03 | Kernel 04 | Change |
|---|---:|---:|---:|
| MIO Throttle (normalized warp stall cycles) | **23.60** | **4.62** | **-80.4%** |
| Warp cycles/issued instruction | 38.44 | **16.23** | **-57.8%** |
| Scheduler cycles with no eligible warp | 79.37% | **75.47%** | -3.90 pp |
| Active warps/scheduler | 7.93 | **3.99** | Lower (register-limited) |
| Eligible warps/scheduler | 1.00 | **0.60** | Lower |
| Memory Pipes Busy | 69.28% | **43.04%** | Lower |

### 4.2 Does the improvement persist at 4096 x 4096?

| Metric | Kernel 03 | Kernel 04 | Change |
|---|---:|---:|---:|
| MIO Throttle (normalized warp stall cycles) | **23.80** | **4.03** | **-83.1%** |
| Warp cycles/issued instruction | 38.99 | **15.72** | **-59.7%** |
| Scheduler cycles with no eligible warp | 79.49% | **74.53%** | -4.96 pp |
| Achieved occupancy | 99.98% | **49.99%** | Approximately halved |
| Standalone throughput | 364.58 | **878.56 GFLOP/s** | **2.41x** |

**The original hypothesis survived the larger test:** normalized MIO stall pressure is much lower even though occupancy is also lower. These MIO values represent *average warp stall cycles per issued instruction*, not aggregate GPU stalled cycles or the time a particular instruction takes.

## 5. Instruction Counts: What Changed Inside the GPU?

Nsight Compute counted dynamically executed warp-level SASS memory instructions:

```cmd
ncu --kernel-name sgemm_block1d --launch-count 1 --metrics smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum block1d_gemm.exe 512 1 0
```

| Instruction type, 512 x 512 | Kernel 03 | Kernel 04 | Change |
|---|---:|---:|---:|
| Shared-memory loads | **5,242,880** | **1,572,864** | **-70.0%** |
| Shared-memory stores | 262,144 | **131,072** | -50.0% |
| Global-memory loads | 270,336 | **139,264** | -48.5% |
| Global-memory stores | 8,192 | **8,192** | Unchanged |

This is the direct measurement we wanted. The new kernel executes **70% fewer shared-memory load instructions** for the same `512 x 512` matrix multiplication, alongside the approximately 80% reduction in normalized MIO Throttle.

The reduction is consistent with register reuse, but it cannot be credited *only* to reuse: we also changed block shape, K-tile size, and the number of threads. SASS instruction counts are **not** byte-transfer counts, so they do not by themselves prove higher DRAM arithmetic intensity.

For completeness, Kernel 04 at `4096 x 4096` executed **805,306,368** shared-memory loads, **67,108,864** shared-memory stores, **67,633,152** global-memory loads, and **524,288** global-memory stores.

## 6. New Question: What Is Stalling Kernel 04 Now?

The detailed `WarpStateStats` output shows a different picture from Kernel 03:

| Warp stall, cycles/issued instruction | K04 at 512 | K04 at 4096 |
|---|---:|---:|
| **MIO Throttle** | **4.62** | **4.03** |
| **Long Scoreboard** | **4.22** | **4.09** |
| Wait | 1.81 | 1.80 |
| Barrier | 1.58 | 1.72 |
| Not Selected | 1.48 | 1.52 |
| Math Pipe Throttle | 0.99 | 1.06 |

At 512, MIO Throttle is still the largest individual category. At 4096, Long Scoreboard is **slightly larger** than MIO Throttle. The difference is too small to call one the unique bottleneck; **both are dominant remaining stalls**.

Long Scoreboard generally means a warp is waiting on an L1TEX-serviced data dependency, frequently a global-memory load. We need source correlation before deciding whether changing the compute tile can address it.

## 7. SourceCounters: Where Are Those Stalls Observed?

A source-level report was collected at `512 x 512` (the `4096 x 4096` collection was too slow to use interactively):

```cmd
ncu --kernel-name sgemm_block1d --launch-count 1 --section SourceCounters --import-source yes -o block1d_source_512 block1d_gemm.exe 512 1 0
ncu --import block1d_source_512.ncu-rep --page source --print-source cuda,sass > block1d_source_512.txt
```

The resulting CUDA/SASS mapping showed:

| Source location | Associated instruction | Main sampled stall |
|---|---|---:|
| Line 97: populate `Bs[tid]` | `STS` shared-memory store dependent on prior `LDG` | **1,121 Long Scoreboard samples** |
| Line 118: load A fragment for compute | `LDS.U.128` shared-memory loads | **1,041 MIO Throttle samples** |
| Line 112: load B operand for compute | `LDS.U` shared-memory loads | **493 MIO Throttle samples** |

**Reading the Long Scoreboard hotspot correctly:** At line 97, the reported wait is attached mostly to an `STS` instruction. The store needs the B value produced by an earlier `LDG` instruction; the sample marks the *waiting instruction*, not the original source of latency. This points to a global-load-to-shared-store dependency while filling the B tile.

**Reading the MIO hotspot correctly:** At lines 112 and 118, samples cluster around generated `LDS` instructions in the compute loop. Even after 1D register reuse, shared-memory instruction pressure remains significant.

SourceCounters is sampled attribution. It supports a performance hypothesis; it does not prove that an individual source line is the complete root cause, and 512-sized source behavior need not exactly match 4096-sized behavior.

## 8. Next Hypothesis: Kernel 05 - 2D Block Tiling

Kernel 04 reuses **B** inside a thread, but a thread still reads eight distinct A values to compute eight outputs at each K step. We can potentially reuse *both* operands by assigning each thread a **small rectangular patch of C** instead of a vertical strip.

### 1D versus 2D work per inner K step

| Conceptual per-thread work | K04: 8 x 1 outputs | Proposed K05: 4 x 4 outputs |
|---|---:|---:|
| A values read from SMEM | 8 | **4** |
| B values read from SMEM | 1 | **4** |
| FMAs | 8 | **16** |
| Scalar SMEM reads per FMA | 1.125 | **0.5** |
| Accumulators per thread | 8 | **16** |

**"2D tiling makes each thread keep a small block of values in registers and reuse them many times. That means fewer loads and more math, so the MIO stall drops."** More precisely, this is our **prediction**: fewer operand loads per FMA should reduce MIO instruction demand, and we will test whether normalized MIO Throttle actually decreases.

A 2D thread tile also creates more independent arithmetic. **More math between loads may help tolerate or partially hide memory latency**, but this does **not** automatically overlap the global loads used to fill the next K-tile. Because the current algorithm loads tiles, synchronizes, and then computes, directly reducing Long Scoreboard may require later software pipelining/prefetching.

### Proposed first experiment (not yet benchmarked)

```cpp
BM = 64; BN = 64; BK = 16;
TM = 4;  TN = 4;
threads_per_block = (BM * BN) / (TM * TN); // 256
```

- `64 x 64` output tile, **256 threads/block** (8 warps); **not more threads** than Kernel 04's 512.
- Each thread owns **16 C outputs**, using a `4 x 4` register accumulator tile.
- Each K step supplies **4 A + 4 B values** to produce **16 FMAs**.
- Shared-memory A/B tiles: `64 x 16` and `16 x 64`, totaling 8 KiB/block.
- Per K-tile, the 256 threads cooperatively load 1,024 elements of A and 1,024 elements of B: four of each per thread on average.

This changes both thread mapping and `BK`, so subsequent tuning should hold parameters fixed where possible to separate the effects. The key checks are: **GFLOP/s, shared-load instructions/FMA, MIO and Long Scoreboard stalls, register allocation, occupancy, and spills**.

## 9. Conclusion

Kernel 04 demonstrates that **register-level B reuse reduced shared-memory instruction demand and MIO pressure enough to more than double throughput on larger matrices despite halving occupancy**. The next bottleneck is not a single clear stall: **MIO Throttle and Long Scoreboard are both substantial**. We will investigate **2D block tiling**, which aims to reuse *both* A and B and do more FMAs per shared-memory read, while treating any reduction in global-load dependency stalls as something to measure rather than assume.
