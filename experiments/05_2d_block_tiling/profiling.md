# Kernel 05 — Nsight Compute Profiling

## 1. Objective

Kernel 05 introduced **2D register tiling** to improve operand reuse during FP32 SGEMM.

Our initial implementation used a 4×4 output patch per thread. After autotuning the output tile, per-thread tile, and K-tile depth, we selected an 8×4 register tile.

Although the optimized configuration achieved higher throughput, we wanted to understand what changed at the GPU execution level.

In particular, we investigated:

- Whether autotuning reduced **MIO Throttle**, a major source of stalls in our earlier GEMM kernels.
- Whether the larger register tile improved warp scheduling and instruction issue efficiency.
- How increased register allocation affected occupancy.
- Whether reduced occupancy necessarily resulted in lower performance.

We profiled only the **initial baseline** and the **best autotuned configuration**. Other candidate configurations are documented through their benchmark results in [results.md](results.md).

---

## 2. Configurations Compared

| Parameter | Baseline 2D | Autotuned 2D |
|---|---|---|
| Source | `block2d_gemm.cu` | `block2d_shapes.cu` |
| Output tile | 64×64 | 128×64 |
| K-tile depth | 8 | 16 |
| Per-thread output tile | 4×4 | 8×4 |
| Outputs/thread | 16 | 32 |
| Threads/block | 256 | 256 |
| Registers/thread (launched specialization) | 64 | 128 |
| Shared memory/block | 4,352 B | 12,800 B |

**GPU:** NVIDIA GeForce GTX 1650 (Turing, Compute Capability 7.5).

**Profiler:** NVIDIA Nsight Compute (`ncu`).

Both implementations use FP32 CUDA cores, shared-memory operand tiles, and register-resident output accumulators.

The tuned implementation increases the number of outputs computed by each thread while also changing the output tile dimensions and K-tile depth.

---

## 3. Profiling at 512×512

We first investigated the kernels on a smaller matrix to compare detailed scheduler, warp-state, memory, and occupancy measurements.

### 3.1 Warp stall comparison

The following stall metrics are normalized warp-stall cycles per issued instruction.

| Metric | Baseline 2D | Autotuned 2D | Change |
|---|---:|---:|---:|
| MIO Throttle | 8.84 | **1.80** | −79.6% |
| Long Scoreboard | 2.42 | **1.50** | −38.0% |
| Barrier | 3.81 | **0.49** | −87.1% |
| Short Scoreboard | 0.98 | **0.11** | −88.8% |
| Warp cycles/issued instruction | 20.38 | **8.45** | −58.5% |

### Observation — MIO Throttle decreased substantially

The baseline kernel spent a large proportion of its normalized warp cycles stalled on MIO Throttle.

After autotuning, MIO Throttle decreased from **8.84 to 1.80**.

This is consistent with the hypothesis that performing more arithmetic per group of shared-memory operands can reduce memory-instruction pipeline pressure.

However, MIO Throttle also includes other operations handled by the MIO pipeline. Without source-level instruction attribution, we cannot conclude that all the reduction came exclusively from shared-memory loads.

### Observation — Barrier stalls also decreased

Barrier stalls decreased from **3.81 to 0.49**.

The tuned kernel uses a deeper K-tile (`BK=16` instead of `BK=8`), requiring fewer tile iterations across the same K dimension.

This provides a plausible explanation for lower synchronization overhead per unit of computation, although the profiler comparison alone cannot isolate that effect from the other changes.

### 3.2 Scheduler statistics

| Metric | Baseline 2D | Autotuned 2D |
|---|---:|---:|
| No eligible warp | 69.36% | **59.91%** |
| Eligible warps/scheduler | 0.63 | **0.88** |
| Issued warps/scheduler/cycle | 0.31 | **0.40** |
| Achieved occupancy | 77.98% | 42.25% |

Despite a substantial reduction in occupancy, the autotuned configuration had more eligible warps per scheduler and higher instruction issue efficiency.

This suggests that the tuned kernel exposes a more favorable balance of independent arithmetic and memory operations.

### 3.3 Register spilling

Nsight Compute reported **zero local-memory spilling requests** for both configurations.

The additional register usage of the tuned kernel therefore did not produce reported register-spill traffic in this profiling experiment.

### Note on small workloads

At 512×512, the baseline launches 64 blocks, while the tuned version launches only 32.

Both execute on a GPU with 14 SMs, so partial waves and the distribution of blocks across SMs can affect achieved occupancy.

For this reason, we also profiled both configurations at 4096×4096.

---

## 4. Profiling at 4096×4096

This was our primary profiling workload because the repeated-run performance benchmarks showed the largest absolute throughput at this matrix size.

### 4.1 Occupancy and scheduler behavior

| Metric | Baseline 2D | Autotuned 2D |
|---|---:|---:|
| Grid size | 64×64 | 64×32 |
| Blocks launched | 4,096 | 2,048 |
| Threads/block | 256 | 256 |
| Theoretical occupancy | 100% | 50% |
| Achieved occupancy | 99.62% | 49.84% |
| Active warps/SM | 31.88 | 15.95 |
| Active warps/scheduler | 7.97 | 3.98 |
| Eligible warps/scheduler | 0.72 | **1.06** |
| Issued warps/scheduler/cycle | 0.31 | **0.43** |
| No eligible warp | 68.58% | **57.10%** |

### Finding 1 — Higher occupancy did not mean better scheduling

The baseline nearly saturated theoretical occupancy, achieving **99.62%**.

The tuned kernel achieved only **49.84%**, because its larger register allocation and shared-memory requirements limit resident blocks.

Yet the tuned kernel had:

- Approximately **47% more eligible warps per scheduler**.
- Approximately **39% higher warp instruction issue rate**.
- Fewer cycles in which the scheduler had no eligible warp.

Although fewer warps were resident, more of them were ready to make progress.

This demonstrates that **occupancy is a resource-utilization constraint, not a direct measure of useful computation or performance**.

The relevant question is whether enough executable instructions are available to keep the GPU's execution pipelines busy.

---

## 5. Warp Stall Analysis at 4096×4096

### 5.1 MIO Throttle

The baseline Nsight Compute report identified MIO Throttle as its dominant reported stall reason:

**Approximately 12.3 normalized stall cycles per issued instruction**, accounting for about 48.4% of its 25.36 warp cycles per issued instruction.

The tuned kernel recorded:

**1.87 normalized stall cycles per issued instruction**, accounting for about 20.2% of its 9.28 warp cycles per issued instruction.

| Metric | Baseline | Tuned |
|---|---:|---:|
| MIO Throttle | ~12.3 | **1.87** |
| Fraction of warp cycles/issued instruction | 48.4% | **20.2%** |
| Relative reduction in MIO stall metric | — | **~85%** |

### Finding 2 — Autotuning substantially relieved MIO pressure

The improvement observed at 512×512 persisted at 4096×4096.

The tuned configuration experienced much less MIO Throttle per issued instruction, despite performing the same overall GEMM operation.

This is strong evidence that our final tile configuration provides a more favorable instruction and operand-reuse pattern than the baseline.

It supports our optimization direction, but does not prove that one individual parameter was responsible. Tile shape, BK, register reuse, thread mapping, and generated instructions all changed together.

### 5.2 Overall warp stall behavior

| Metric | Baseline | Tuned |
|---|---:|---:|
| Warp cycles/issued instruction | 25.36 | **9.28** |
| No eligible warp | 68.58% | **57.10%** |
| Issued warps/scheduler/cycle | 0.31 | **0.43** |

Normalized warp cycles per issued instruction decreased by approximately **63.4%**.

The tuned kernel also had fewer scheduler cycles with no eligible warp.

These findings are consistent with improved ability to execute useful instructions despite lower active-warp occupancy.

### 5.3 Other tuned-kernel stalls

The detailed 4096×4096 warp-state report for the selected kernel showed:

| Stall reason | Tuned value |
|---|---:|
| MIO Throttle | 1.87 |
| Long Scoreboard | 1.84 |
| Not Selected | 1.46 |
| Math Pipe Throttle | 1.32 |
| Wait | 1.01 |
| Barrier | 0.48 |
| Short Scoreboard | 0.12 |

The tuned kernel's largest reported stall categories were **MIO Throttle and Long Scoreboard**, with comparable normalized values.

This suggests further optimization opportunities may remain. However, a higher value in another stall category does not necessarily identify the next profitable optimization.

The baseline 4096 report did not include the complete numeric stall table, so we do not claim an exact baseline-to-tuned Long Scoreboard reduction at this matrix size.

---

## 6. Connecting Profiling to Performance

Repeated-run benchmark results:

| Kernel | 4096×4096 median time | Throughput |
|---|---:|---:|
| Baseline 2D | 96.7864 ms | 1,420.0230 GFLOP/s |
| **Autotuned 2D** | **72.1797 ms** | **1,904.1217 GFLOP/s** |

**Improvement: approximately 1.34×.**

Individual Nsight Compute runs also showed the tuned kernel outperforming the baseline, but profiler-instrumented timings are not substituted for the repeated-run benchmark results.

The hardware measurements help explain the performance improvement:

1. Normalized MIO Throttle decreased substantially.
2. Synchronization-related stalls decreased in the 512×512 comparison.
3. Scheduler issue efficiency improved.
4. Fewer resident warps were needed to maintain a larger pool of eligible instructions.
5. Increased register allocation did not cause reported spilling.

These improvements collectively provide a plausible execution-level explanation for the higher measured throughput.

We cannot assign a precise fraction of the speedup to each factor without additional controlled experiments.

---

## 7. Reproducing the Profiling

Run these commands from `experiments/05_2d_block_tiling/` in the Visual Studio 2022 x64 Native Tools Command Prompt.

### Baseline — 512×512

```cmd
ncu --kernel-name regex:sgemm_block2d --launch-count 1 --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --print-details all block2d_gemm.exe 512 1 0
```

### Autotuned — 512×512

```cmd
ncu --launch-count 1 --section LaunchStats --section Occupancy --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --print-details all block2d_best.exe 512 1 0
```

### Baseline — 4096×4096

```cmd
ncu --launch-count 1 --section Occupancy --section SchedulerStats --section WarpStateStats block2d_gemm.exe 4096 1 0
```

### Autotuned — 4096×4096

```cmd
ncu --launch-count 1 --section Occupancy --section SchedulerStats --section WarpStateStats block2d_best.exe 4096 1 0
```

### Detailed tuned warp stalls — 4096×4096

```cmd
ncu --launch-count 1 --section WarpStateStats --print-details all block2d_best.exe 4096 1 0
```

The profiler performs multiple collection passes, which can affect execution timing and cache behavior.

Detailed performance claims should therefore use the independent repeated-run CUDA-event benchmarks.

---

## 8. Final Conclusions

Our Kernel 05 investigation began with the idea that **2D register tiling could increase reuse of both A and B operands**.

Autotuning revealed that an asymmetric 8×4 thread tile within a 128×64×16 block tile performed better than the initial 4×4 implementation.

Nsight Compute then provided evidence of why the selected configuration was more effective.

The key findings were:

- **MIO Throttle fell by approximately 85%** at 4096×4096, measured in normalized stall cycles per issued instruction.
- **Warp cycles per issued instruction decreased by approximately 63%.**
- **Scheduler instruction issue efficiency improved by approximately 39%.**
- **Achieved occupancy decreased from approximately 100% to 50%.**
- **No register spilling was reported.**
- **Repeated-run performance improved by approximately 1.34× over the initial 2D kernel.**

### Central lesson

> More active warps do not automatically produce a faster GPU kernel. A carefully selected register tile can improve instruction readiness, operand reuse, and execution efficiency enough to outperform a higher-occupancy configuration.

**Kernel 05 is now profiled and its performance behavior is documented.**

The remaining task is to archive the raw Nsight Compute outputs in the separate profiling repository before continuing to the next GEMM optimization.