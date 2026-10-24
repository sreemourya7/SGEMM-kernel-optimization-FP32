# Kernel 03 — Nsight Compute Profiling

[View Raw Profiling Metrics](https://github.com/sreemourya7/cuda-nsight-compute-profiling/blob/main/kernels/03_smem_cuda/README.md)

## What Changed?

Kernel 03 introduced shared-memory tiling to reduce repeated global-memory loads. However, performance improved by only **~1.1×** over Kernel 02.

Let's investigate why.

## What Nsight Shows

Comparing both kernels at `512 × 512`:

| Metric | Kernel 02 | Kernel 03 |
|---|---:|---:|
| SM Throughput | 86.47% | 69.13% |
| Achieved Occupancy | 98.52% | 99.78% |
| Eligible Warps / Scheduler | 1.13 | 1.00 |
| No Eligible Warp | 68.54% | 79.37% |
| Dominant Stall | LG Throttle | MIO Throttle |
| Mem Pipes Busy | 86.02% | 69.28% |

Occupancy remained excellent, but scheduler eligibility worsened.

Our previous bottleneck was **LG Throttle**. After introducing shared memory, Nsight instead reports **MIO Throttle** as the dominant stall reason.

## Investigating 4096 × 4096

| Metric | Value |
|---|---:|
| Occupancy | 99.98% |
| No Eligible Warp | 79.49% |
| MIO Throttle | 23.8 cycles |
| Warp Cycles / Issued Instruction | 38.99 |
| DRAM Throughput | 13.97% |

The same behavior persists at larger matrix sizes.

Approximately **61.1% of the average warp issue interval** is attributed to MIO Throttle.

The MIO pipeline handles shared-memory instructions, among other operations. Our inner loop repeatedly loads values from shared memory:

```cpp
tmp += As[ty][k] * Bs[k][tx];
```

These frequent shared-memory accesses are a likely contributor to MIO queue pressure, although further instruction-level analysis is needed to confirm the root cause.

## Next Optimization — 1D Block Tiling

Currently, each thread computes only **one element of C**.

With 1D block tiling, each thread computes multiple output elements, maintaining their partial sums in registers.

```text
Kernel 03
One output per thread
        ↓
Repeated shared-memory loads
        ↓
MIO pipeline pressure


Kernel 04 — 1D Block Tiling
Multiple outputs per thread
        ↓
Reuse shared-memory values
        ↓
Accumulate results in registers
        ↓
Fewer shared-memory loads per output
        ↓
Potentially lower MIO Throttle
```

**Hypothesis:** Increasing per-thread data reuse through 1D block tiling should reduce shared-memory instruction pressure and improve compute throughput.