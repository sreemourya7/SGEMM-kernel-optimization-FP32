# Kernel 01 — Nsight Compute Profiling

**GPU:** GTX 1650 | **Profiler:** Nsight Compute 2025.4.1

[Raw Profiling Metrics](https://github.com/sreemourya7/cuda-nsight-compute-profiling/blob/main/kernels/01_naive_cuda/README.md)

## Goal

Benchmarking tells us **how fast** our kernel runs. Profiling helps explain **why**.

## Five Questions to Ask When Profiling

| # | Question | Nsight Section |
|---|---|---|
| 1 | Is the GPU doing useful work? | GPU Speed of Light |
| 2 | Do we have enough resident warps? | Occupancy |
| 3 | Are those warps ready to execute? | Scheduler Statistics |
| 4 | Why are the warps stalled? | Warp State Statistics |
| 5 | What hardware behavior causes the bottleneck? | Memory Workload Analysis |

## Investigating Kernel 01

### 1. Is the GPU doing useful work?

**GPU Speed of Light:**

- SM Throughput: **5.86%**
- Memory Throughput: **48.32%**

Compute utilization is low, while memory utilization is considerably higher. Let's investigate why.

### 2. Do we have enough resident warps?

**Occupancy:**

- Achieved Occupancy: **98.28%**
- Active Warps/SM: **31.45 / 32**

Almost all available warp slots are occupied. Insufficient occupancy is unlikely to be our problem.

But **resident doesn't mean executing**.

### 3. Are those warps ready to execute?

**Scheduler Statistics:**

- Active Warps/Scheduler: **7.86**
- Eligible Warps/Scheduler: **0.08**
- No Eligible Warp: **97.87%**

Nearly 8 warps are resident per scheduler, but almost none are ready to issue instructions.

**Question:** What makes these warps wait?

### 4. Why are the warps stalled?

**Warp State Statistics:**

- Warp Cycles per Issued Instruction: **369.19**
- LG Throttle: **353.8 cycles**

Approximately 95.8% of the average warp cycles between issued instructions are attributed to LG Throttle.

**LG Throttle:** A warp cannot issue its next local/global memory operation because the corresponding L1 instruction queue is full.

**Question:** Why is the queue backing up?

### 5. What hardware behavior causes the bottleneck?

**GPU Speed of Light + Memory Workload Analysis:**

- L1/TEX Throughput: **96.63%**
- DRAM Throughput: **0.76%**
- L1/TEX Hit Rate: **99.21%**

The L1/TEX path is near its reported peak throughput, while DRAM bandwidth utilization is very low.

**Diagnosis:** High occupancy → Few eligible warps → LG Throttle → L1/TEX processing pressure → Low compute utilization.

## Why Might Memory Coalescing Help?

### 1. Memory sectors

A warp contains 32 threads. Each thread loads a 4-byte float.

The GPU services global-memory requests using **32-byte sectors**.

For one warp-level load:

| Access pattern | Sectors needed |
|---|---:|
| 32 scattered floats (different sectors) | 32 |
| 32 consecutive, aligned floats | 4 |
| All 32 lanes load the same float | 1 |

**Coalescing reduces sectors accessed, not necessarily load instructions executed.**

### 2. Wavefronts

The L1/TEX unit processes memory requests in units called **wavefronts**.

More sectors can require more wavefronts and more processing work. However, sectors and wavefronts do not have a fixed 1:1 relationship.

### 3. Our naive kernel vs a coalesced kernel

For each iteration of the inner product loop:

| Warp-level load | Naive mapping | Proposed mapping |
|---|---|---|
| A | 32 scattered sectors | 1 sector (broadcast) |
| B | 1 sector (broadcast) | 4 adjacent sectors |
| **Total** | **~33 sectors** | **~5 sectors** |

*Illustrative counts for aligned, full-warp accesses; these are not measured profiling results.*

### 4. How this could reduce LG Throttle

**Current kernel:**

Scattered loads → More sectors → Potentially more L1/TEX work → Queue pressure → LG Throttle → Warps stall.

**Coalesced kernel:**

Contiguous/broadcast loads → Fewer sectors → Potentially less L1/TEX work → Queue drains more efficiently → Fewer stalls.

### Optimization Hypothesis

**Remapping threads for coalesced memory access should reduce memory-request processing pressure, improve warp eligibility, and increase GEMM throughput.**

We'll test this hypothesis in **Kernel 02** using benchmarks and Nsight Compute.