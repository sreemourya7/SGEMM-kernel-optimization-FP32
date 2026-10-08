# CPU Baseline — Benchmark Results

This document records the benchmark configuration, methodology, raw measurements, and final results for the naive FP32 CPU implementation of SGEMM.

The purpose of this benchmark is to establish a reproducible CPU baseline before moving the same computation to CUDA.

---

## System Configuration

### Software

| Component | Configuration |
|---|---|
| Compiler | Microsoft MSVC 19.44 |
| Optimization | `/O2` |
| C++ Standard | C++17 |
| Floating-Point Mode | Default MSVC mode (`/fp:precise`) |
| CUDA Toolkit | CUDA 13.1 |
| Operating System | Windows |

### Hardware

| Component | Configuration |
|---|---|
| CPU | Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz |
| GPU | NVIDIA GeForce GTX 1650 |
| GPU Architecture | Turing |
| Precision Benchmarked | FP32 |

### Build Command

```cmd
cl /O2 /EHsc /std:c++17 cpu_gemm.cpp /Fe:cpu_gemm.exe
```

No fast-math option was enabled.

---

## Benchmark Methodology

For each matrix size:

1. Generate FP32 matrices `A` and `B` using a fixed random seed of `42`.
2. Compute a reference result using FP64 accumulation.
3. Execute one untimed FP32 GEMM as a warm-up.
4. Verify the complete warm-up result against the FP64 reference.
5. Execute three timed FP32 GEMM runs.
6. Verify the complete output matrix after every timed run.
7. Report the median runtime and median achieved GFLOP/s as the official benchmark result.

The amount of work for an `N × N` matrix multiplication is approximated as:

$$
\text{FLOPs} \approx 2N^3
$$

Performance is calculated as:

$$
\text{GFLOP/s} =
\frac{2N^3}
{\text{execution time} \times 10^9}
$$

---

# Final Benchmark Results

| Matrix Size | Total Work | Median Time | Median Performance | Best Performance | Verification |
|---:|---:|---:|---:|---:|:---:|
| `512 × 512` | `0.2684 GFLOPs` | `0.2101 s` | **1.2774 GFLOP/s** | `1.3024 GFLOP/s` | PASS |
| `1024 × 1024` | `2.1475 GFLOPs` | `1.6671 s` | **1.2882 GFLOP/s** | `1.3235 GFLOP/s` | PASS |
| `1536 × 1536` | `7.2478 GFLOPs` | `11.2495 s` | **0.6443 GFLOP/s** | `0.6503 GFLOP/s` | PASS |
| `2000 × 2000` | `16.0000 GFLOPs` | `38.4521 s` | **0.4161 GFLOP/s** | `0.4278 GFLOP/s` | PASS |
| `2048 × 2048` | `17.1799 GFLOPs` | `123.3229 s` | **0.1393 GFLOP/s** | `0.1470 GFLOP/s` | PASS |

The **median result** is used throughout the main README when comparing implementations.

---

# Raw Measurements

## 512 × 512

Total work:

$$
0.2684\ \text{GFLOPs}
$$

| Run | Execution Time | Performance | Verification |
|---:|---:|---:|:---:|
| 1 | `0.2101 s` | `1.2774 GFLOP/s` | PASS |
| 2 | `0.2061 s` | **1.3024 GFLOP/s** | PASS |
| 3 | `0.2220 s` | `1.2093 GFLOP/s` | PASS |

**Median:** `0.2101 s` — **1.2774 GFLOP/s**

**Best observed:** `0.2061 s` — **1.3024 GFLOP/s**

---

## 1024 × 1024

Total work:

$$
2.1475\ \text{GFLOPs}
$$

| Run | Execution Time | Performance | Verification |
|---:|---:|---:|:---:|
| 1 | `1.6913 s` | `1.2697 GFLOP/s` | PASS |
| 2 | `1.6226 s` | **1.3235 GFLOP/s** | PASS |
| 3 | `1.6671 s` | `1.2882 GFLOP/s` | PASS |

**Median:** `1.6671 s` — **1.2882 GFLOP/s**

**Best observed:** `1.6226 s` — **1.3235 GFLOP/s**

---

## 1536 × 1536

Total work:

$$
7.2478\ \text{GFLOPs}
$$

| Run | Execution Time | Performance | Verification |
|---:|---:|---:|:---:|
| 1 | `11.2859 s` | `0.6422 GFLOP/s` | PASS |
| 2 | `11.2495 s` | `0.6443 GFLOP/s` | PASS |
| 3 | `11.1456 s` | **0.6503 GFLOP/s** | PASS |

**Median:** `11.2495 s` — **0.6443 GFLOP/s**

**Best observed:** `11.1456 s` — **0.6503 GFLOP/s**

---

## 2000 × 2000

Total work:

$$
16.0000\ \text{GFLOPs}
$$

| Run | Execution Time | Performance | Verification |
|---:|---:|---:|:---:|
| 1 | `41.9274 s` | `0.3816 GFLOP/s` | PASS |
| 2 | `38.4521 s` | `0.4161 GFLOP/s` | PASS |
| 3 | `37.4013 s` | **0.4278 GFLOP/s** | PASS |

**Median:** `38.4521 s` — **0.4161 GFLOP/s**

**Best observed:** `37.4013 s` — **0.4278 GFLOP/s**

---

## 2048 × 2048

Total work:

$$
17.1799\ \text{GFLOPs}
$$

| Run | Execution Time | Performance | Verification |
|---:|---:|---:|:---:|
| 1 | `136.1601 s` | `0.1262 GFLOP/s` | PASS |
| 2 | `123.3229 s` | `0.1393 GFLOP/s` | PASS |
| 3 | `116.8995 s` | **0.1470 GFLOP/s** | PASS |

**Median:** `123.3229 s` — **0.1393 GFLOP/s**

**Best observed:** `116.8995 s` — **0.1470 GFLOP/s**

---

# Notable Observation: 2000 vs. 2048

The `2000 × 2000` and `2048 × 2048` cases perform a similar amount of arithmetic:

| Matrix Size | Work | Median Time | Median Performance |
|---:|---:|---:|---:|
| `2000 × 2000` | `16.0000 GFLOPs` | `38.4521 s` | `0.4161 GFLOP/s` |
| `2048 × 2048` | `17.1799 GFLOPs` | `123.3229 s` | `0.1393 GFLOP/s` |

The `2048 × 2048` case performs only about 7% more arithmetic, yet requires more than three times the execution time.

The naive `i-j-k` implementation accesses matrix `B` using:

```cpp
B[k * N + j]
```

For `N = 2048`, consecutive accesses are separated by:

$$
2048 \times 4\ \text{bytes} = 8192\ \text{bytes}
$$

This power-of-two stride may interact poorly with the CPU cache and memory hierarchy. The result is retained as an interesting performance characteristic rather than being discarded as an outlier.

---

# Best Observed CPU Result

The highest measured throughput from this baseline implementation was:

> **1.3235 GFLOP/s**

for a `1024 × 1024` matrix multiplication.

The official median result for the same size was:

> **1.2882 GFLOP/s**

These CPU measurements will serve as the baseline for subsequent CUDA implementations.
