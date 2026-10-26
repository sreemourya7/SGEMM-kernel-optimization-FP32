# Kernel 03 — Nsight Compute Profiling

**GPU:** NVIDIA GTX 1650 (Turing, SM 7.5)  
**Kernel:** `sgemm_smem` — 32×32 Shared-Memory Tiling  
**Profiler:** NVIDIA Nsight Compute (CLI)

[Complete Raw Profiling Metrics](https://github.com/sreemourya7/cuda-nsight-compute-profiling/blob/main/kernels/03_smem_cuda/README.md)

## 1. The Problem — Why Was Shared-Memory Tiling Only Slightly Faster?

Kernel 02 introduced global-memory coalescing. Kernel 03 extended this by loading tiles of A and B into shared memory, allowing threads within a block to reuse data.

We expected substantially better performance. Instead:

| 512×512 benchmark | Kernel 02 | Kernel 03 |
|---|---:|---:|
| Performance | 258.52 GFLOP/s | 280.63 GFLOP/s |
| Improvement | — | **1.09×** |

Shared-memory tiling worked, but delivered only a modest speedup.

**Question:** What is preventing Kernel 03 from becoming significantly faster?

## 2. 1st Profiling Pass — Overall GPU Performance

**Command 1:**

```cmd
ncu --kernel-name sgemm_smem --launch-count 1 smem_gemm.exe 512 1 0
```

This launches Nsight Compute (`ncu`) to profile one matching launch of `sgemm_smem`.

- `--kernel-name sgemm_smem`: Select our CUDA kernel.
- `--launch-count 1`: Profile one matching kernel launch (Nsight may replay it internally).
- `smem_gemm.exe`: Our CUDA program.
- `512 1 0`: Matrix dimension, benchmark repetitions, and verification flag.

**Key results at 512×512:**

| Metric | Value |
|---|---:|
| SM Throughput | 69.13% |
| L1/TEX Throughput | 81.97% |
| DRAM Throughput | 10.71% |
| Achieved Occupancy | 99.78% |
| Registers/thread | 44 |

Occupancy is almost 100%, yet performance remains modest.

This suggests that simply making more warps resident is unlikely to solve our problem. High occupancy does **not** mean warps are continuously ready to issue instructions.

We need to inspect warp scheduling and stalls.

## 3. 2nd Profiling Pass — Why Are Warps Stalling?

**Command 2:**

```cmd
ncu --kernel-name sgemm_smem --launch-count 1 --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis smem_gemm.exe 512 1 0
```

`--section` requests a predefined group of related metrics.

| Section | Question |
|---|---|
| `SchedulerStats` | Are enough warps ready to issue instructions? |
| `WarpStateStats` | Why are warps unable to issue instructions? |
| `MemoryWorkloadAnalysis` | How busy are the memory pipelines? |

**Results:**

| Metric | Value |
|---|---:|
| Active warps/scheduler | 7.93 |
| Eligible warps/scheduler | 1.00 |
| No eligible warp | 79.37% |
| Warp cycles/issued instruction | 38.44 |
| Dominant stall | **MIO Throttle — 23.6 cycles** |
| Memory Pipes Busy | 69.28% |

Despite excellent occupancy, a scheduler has **no eligible warp during approximately 79% of cycles**.

The dominant stall reason is **MIO Throttle**.

MIO (Memory Input/Output) Throttle occurs when the MIO instruction queue is full. This queue services instructions including shared-memory operations, special math, and certain control operations.

Approximately 61.4% of the average warp issue interval is attributed to MIO Throttle.

### What Changed From Kernel 02?

| Metric | Kernel 02 | Kernel 03 |
|---|---:|---:|
| Achieved Occupancy | 98.52% | 99.78% |
| No eligible warp | 68.54% | 79.37% |
| Dominant stall | LG Throttle | **MIO Throttle** |
| Memory Pipes Busy | 86.02% | 69.28% |

Kernel 02 experienced pressure on the local/global memory instruction queue (LG Throttle).

After introducing shared-memory tiling, **the dominant stall shifted to MIO Throttle**.

Shared-memory instructions are now a leading suspect, but we need more evidence.

## 4. 3rd Profiling Pass — Count Memory Instructions

**Command 3:**

```cmd
ncu --kernel-name sgemm_smem --launch-count 1 --metrics smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum smem_gemm.exe 512 1 0
```

Unlike `--section`, which selects a group of metrics, `--metrics` requests specific named counters.

These counters measure dynamically executed **warp-level GPU machine instructions (SASS)**, not bytes transferred.

| Instruction category | Executed instructions |
|---|---:|
| Shared-memory loads | **5,242,880** |
| Shared-memory stores | 262,144 |
| Global-memory loads | 270,336 |
| Global-memory stores | 8,192 |

The kernel executes approximately **19.4× as many shared-memory load instructions as global-memory load instructions**.

This is consistent with an inner loop that repeatedly loads A and B values from shared memory.

However, a large instruction count alone doesn't establish which instructions are associated with MIO stalls.

**Next question:** Where in our kernel are these stalls observed?

## 5. 4th Profiling Pass — Trace Stalls Back to Source Code

### Step A — Compile With Source-Line Mapping

**Command 4:**

```cmd
nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo smem_gemm.cu -o smem_gemm.exe
```

The important option is `-lineinfo`.

It lets Nsight associate generated GPU assembly instructions with their original CUDA source lines, without disabling normal compiler optimizations.

### Step B — Collect Source-Level Stall Data

**Command 5:**

```cmd
ncu --kernel-name sgemm_smem --launch-count 1 --section SourceCounters --import-source yes -o smem_source_512 smem_gemm.exe 512 1 0
```

- `SourceCounters`: Collects instruction-associated stall samples (**IMPORTANT**).
- `--import-source yes`: Includes available CUDA source code in the report.
- `-o smem_source_512`: Saves the report as `smem_source_512.ncu-rep`.

### Step C — Read the Report

**Command 6:**

```cmd
ncu --import smem_source_512.ncu-rep --page source --print-source cuda,sass
```

- `--import`: Reads an existing profiling report rather than executing the kernel again.
- `--page source`: Displays source-level profiling information.
- `--print-source cuda,sass`: Shows CUDA source and GPU assembly instructions.

### What Did We Discover?

Nsight identified line 68:

```cpp
tmp += As[ty][k] * Bs[k][tx];
```

The compiler generated instructions including:

| SASS instruction | Meaning |
|---|---|
| `LDS.U` | Shared-memory load |
| `LDS.U.128` | 128-bit shared-memory load |
| `FFMA` | Floating-point fused multiply-add |

**SourceCounters results for line 68:**

| Metric | Value |
|---|---:|
| Total warp-state samples | 8,503 |
| MIO Throttle samples | **6,788** |
| MIO share of line samples | ~79.8% |

The MIO Throttle samples on this line were associated with `LDS` instructions, rather than the `FFMA` instructions.

The compiler's unrolled inner loop contained:

- **32 scalar shared-memory load instructions**
- **8 vectorized 128-bit shared-memory load instructions**
- **32 FFMA instructions**

The 40 load instructions execute once for each warp's traversal of one K-tile, contributing to the millions of shared-memory load instructions measured earlier.

This is stronger evidence than simply observing a large load count.

**Diagnosis:** Frequent shared-memory load instructions in our compute loop are strongly associated with MIO instruction-queue pressure.

SourceCounters uses sampling, so this establishes a well-supported bottleneck hypothesis rather than exact per-instruction causation.

## 6. Does the Same Problem Exist at 4096×4096?

We repeated Commands 1 and 2 with the matrix argument changed from `512` to `4096`.

| Metric | 512×512 | 4096×4096 |
|---|---:|---:|
| Achieved Occupancy | 99.78% | 99.98% |
| No eligible warp | 79.37% | 79.49% |
| MIO Throttle | 23.6 cycles | 23.8 cycles |
| Warp cycles/issued instruction | 38.44 | 38.99 |
| DRAM Throughput | 10.71% | 13.97% |

The scheduling and stall behavior remains remarkably similar.

Therefore, MIO Throttle is not merely an artifact of profiling a small matrix. It persists at our larger workload.

## 7. Optimization Hypothesis — Why Compute Multiple C Elements Per Thread?

Our existing kernel computes **one output element per thread**.

At each K iteration:

```cpp
tmp += As[ty][k] * Bs[k][tx];
```

Conceptually, this requires:

- 1 shared-memory A value
- 1 shared-memory B value
- 1 FMA

That is **2 scalar loads per FMA** before accounting for compiler optimizations such as vectorization.

### Proposed Kernel 04 — 1D Block Tiling

Instead, let one thread compute eight vertically adjacent output elements:

```text
C[0][col]
C[1][col]
C[2][col]
C[3][col]
C[4][col]
C[5][col]
C[6][col]
C[7][col]
```

For one K iteration, the thread needs:

- 8 distinct A values
- **1 B value shared across all eight computations**
- 8 FMAs, accumulated into 8 registers

| Per K iteration | One output/thread | Eight outputs/thread |
|---|---:|---:|
| A loads | 1 | 8 |
| B loads | 1 | 1 |
| FMAs | 1 | 8 |
| **Scalar loads/FMA** | **2.0** | **1.125** |

This is a theoretical, per-thread comparison. Actual SASS instruction counts may differ because of vectorization and compiler scheduling.

### Why Might This Relieve MIO Throttle?

```text
Kernel 03 — One output per thread
Repeated loads of A and B from shared memory
                    ↓
High shared-memory instruction demand
                    ↓
MIO instruction-queue pressure


Kernel 04 — Eight outputs per thread
Load B once → reuse across eight FMAs
                    ↓
Fewer shared-memory loads per FMA
                    ↓
Potentially reduced MIO queue pressure
                    ↓
More opportunities for warps to issue instructions
                    ↓
Potentially higher GFLOP/s
```

The goal is to increase **register-level data reuse**, reducing the number of shared-memory load instructions required for the same mathematical work.

This is not necessarily a reduction in DRAM traffic or an improvement in DRAM-level arithmetic intensity.

### The Tradeoff

Computing eight outputs requires eight register accumulators per thread.

Higher register usage may reduce occupancy or introduce other bottlenecks. Thus, lower MIO Throttle and higher performance are **hypotheses to test**, not guaranteed outcomes.

## 8. Next Experiment

Implement Kernel 04 using **1D Block Tiling (`TM=8`)**.

Compare it against Kernel 03 using:

1. Standalone GFLOP/s and execution time.
2. Executed shared-memory load instructions relative to useful computation.
3. MIO Throttle and scheduler eligibility.
4. Register usage and achieved occupancy.

**Final takeaway:** Profiling revealed that high occupancy alone was insufficient. We traced the dominant MIO stalls from scheduler statistics to shared-memory load instructions in the GEMM inner loop. This provides a concrete, hardware-informed reason to investigate register reuse through 1D block tiling.