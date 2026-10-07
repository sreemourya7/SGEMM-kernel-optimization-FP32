# Experiment 00 — Why GPUs for Matrix Multiplication?

Before writing a single CUDA kernel, I wanted to understand the problem we are trying to accelerate.

Why does matrix multiplication matter so much?  
How expensive is it on a straightforward CPU implementation?  
And what makes GPUs particularly good at this kind of computation?

Let's start from stratch.

---

## 1. What is GEMM?

**GEMM** stands for **General Matrix Multiplication**. The general operation is:

\[
C = α*AB + β*C
\]

where `A`, `B`, and `C` are matrices and `α` and `β` are scalar values. For now, lets focus on the simpler case:

\[
C = AxB
\]

Before understanding how to optimize GEMM, lets understand where exactly is Matrix multiplication used and why it is one of the fundamental computational building blocks behind modern machine learning.

Operations inside:

- linear layers,
- attention,
- MLPs,
- projections,
- convolutions after suitable transformations,

these eventually involve enormous amounts of matrix multiplication.

That makes GEMM an excellent workload for learning GPU performance engineering. If we understand how to make GEMM fast, we start understanding many of the same ideas used to make modern AI systems fast.

---

## 2. How does `A × B = C` actually work?

Consider:

```text
A                     B

[ a00 a01 a02 ]       [ b00 b01 ]
[ a10 a11 a12 ]   ×   [ b10 b11 ]
                      [ b20 b21 ]
```

The resulting matrix `C` will be:

```text
[ c00 c01 ]
[ c10 c11 ]
```

Each element of `C` is produced by taking:

**one row from A** and computing its dot product with **one column from B**
For example:

```text
c00 = a00*b00 + a01*b10 + a02*b20
```

In general:

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik}B_{kj}
$$

The interesting part is that `C[0][0]`, `C[0][1]`, `C[1][0]` and so on can largely be computed independently. It's worth keeping this thought in mind. It is going to become very important when we move to the GPU.

---

## 3. Why does GEMM require approximately `2N³` FLOPs?

For two square `N × N` matrices, the result contains `N²` elements. To compute each output element, we perform approximately `N` iterations of the innermost loop:

```cpp
sum += A[i * N + k] * B[k * N + j];
```

Each iteration performs one floating-point multiplication and one floating-point addition, so we count approximately `2N` floating-point operations per output element. Since there are `N²` output elements, the total work is:

$$
N^2 \times 2N = 2N^3
$$

Therefore, the classical matrix multiplication algorithm performs approximately:

$$
\boxed{\text{FLOPs} \approx 2N^3}
$$

For example, for a `4096 × 4096` matrix multiplication:

$$
2(4096)^3 \approx 137.4 \text{ billion FLOPs}
$$

So those three innocent-looking loops are responsible for roughly **137 billion floating-point operations**. Matrix sizes grow quickly, and the amount of computation grows even faster, which is exactly why extracting as much hardware performance as possible starts to matter.

---

## 4. Starting with the Simplest CPU Implementation

Before moving to CUDA, I first wanted a simple baseline that I could measure and compare against later GPU implementations. The most direct way to implement matrix multiplication on the CPU is to follow the mathematical definition almost exactly:

```cpp
for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;

        for (int k = 0; k < N; ++k) {
            sum += A[i * N + k] * B[k * N + j];
        }

        C[i * N + j] = sum;
    }
}
```

Here, `i` selects a row of `A`, `j` selects a column of `B`, and `k` walks through both to compute their dot product. So for every output element `C[i][j]`, the program takes **row `i` of A** and **column `j` of B**, multiplies the corresponding elements, accumulates the result, and stores it in `C`.

This implementation is intentionally simple. There is no tiling, vectorization, multithreading, cache blocking, or other CPU-specific optimization. It is also not meant to represent the performance of highly optimized CPU libraries such as Intel MKL or OpenBLAS.

The purpose of this version is simply to establish a clean starting point: **how fast is the most straightforward implementation before introducing GPU parallelism and progressively optimizing memory access and computation?** This baseline gives us something concrete to compare against as the CUDA kernels become more sophisticated.

👉 [View the full CPU implementation](./experiments/00_cpu_baseline/cpu_gemm.cpp)

---

## 5. How Fast Is My CPU Implementation?

Execution time alone does not tell us much about performance. A 2-second runtime means very different things depending on whether the program performed one million or one trillion operations.

Since GEMM performs approximately \(2N^3\) floating-point operations, we can measure throughput in GFLOP/s:

$$
\text{GFLOP/s} =
\frac{2N^3}
{\text{execution time} \times 10^9}
$$

I will benchmark several matrix sizes and record both execution time and achieved throughput.

| Matrix Size | Execution Time | Performance |
|---:|---:|---:|
| `512 × 512` | TBD | TBD GFLOP/s |
| `1024 × 1024` | TBD | TBD GFLOP/s |
| `2048 × 2048` | TBD | TBD GFLOP/s |
| `4096 × 4096` | TBD | TBD GFLOP/s |

These values will be replaced with measurements from my machine.

👉 [View detailed benchmark results](./experiments/00_cpu_baseline/results.md)

The main number I care about is **achieved GFLOP/s**. This gives us a common metric that can later be used to compare the CPU baseline, our CUDA kernels, and the theoretical capability of the GPU.

---

## 6. How Does This Compare with My GTX 1650?

My GPU is an **NVIDIA GeForce GTX 1650**, with theoretical FP32 throughput of roughly:

$$
3\ \text{TFLOP/s} \approx 3000\ \text{GFLOP/s}
$$

Peak FP32 throughput can be estimated roughly as:

$$
\text{CUDA cores}
\times
\text{clock frequency}
\times
2
$$

The factor of `2` comes from a fused multiply-add (FMA):

```text
a * b + c
```

which performs one multiplication and one addition.

The important distinction is that **theoretical peak performance is not the same as achieved performance**. A GPU capable of roughly 3 TFLOP/s does not automatically make every CUDA kernel run anywhere near that speed. Poor memory access, insufficient parallelism, synchronization overhead, or inefficient use of the execution units can leave much of the hardware underutilized.

This gap between **what the GPU can theoretically provide** and **what our kernel actually achieves** is the central problem of GPU performance engineering. The goal of the following experiments is to progressively close that gap.

---

## 7. Why Is GEMM So Well Suited to GPUs?

Consider the output matrix:

```text
C[0][0]   C[0][1]   C[0][2]   ...
C[1][0]   C[1][1]   C[1][2]   ...
C[2][0]   C[2][1]   C[2][2]   ...
...
```

Each output element performs essentially the same operation: a dot product between one row of `A` and one column of `B`. More importantly, many of these output elements can be computed independently.

This gives GEMM a large amount of **data parallelism**.

CPUs typically contain a smaller number of powerful cores optimized for general-purpose workloads, low-latency execution, complex control flow, and branch-heavy programs. GPUs instead dedicate much more hardware to executing large numbers of similar arithmetic operations concurrently.

GEMM maps naturally to this design because most of its work consists of repeated multiply-add operations across a large number of independent output elements. This is also one of the reasons matrix multiplication is such a fundamental workload in modern machine learning and deep learning systems.

---

## 8. Isn't There a Mathematically Faster Matrix Multiplication Algorithm?

The classical matrix multiplication algorithm has complexity:

$$
O(N^3)
$$

Algorithms with better asymptotic complexity do exist. For example, **Strassen's algorithm** reduces the complexity to approximately:

$$
O(N^{2.807})
$$

and later algorithms improve the theoretical exponent even further.

However, Big-O complexity is only part of the performance story. Real hardware performance also depends on memory access patterns, cache behavior, memory bandwidth, data reuse, synchronization, register usage, occupancy, instruction throughput, and parallelism.

An algorithm may perform fewer arithmetic operations on paper while still running slower for practical matrix sizes because of larger constants or poor interaction with the hardware.

This leads to an important idea that will appear throughout this project:

> **The fastest algorithm on paper is not necessarily the fastest implementation on real hardware.**

GPU performance engineering is therefore not only about reducing the number of operations. It is about understanding **how efficiently the hardware can execute those operations**.

---

## Next: Naive CUDA GEMM

At this point, we have a mathematical understanding of GEMM, a simple CPU baseline, a way to measure throughput, and a rough idea of the GPU's theoretical capability.

The next step is to move the same computation onto the GPU with the simplest CUDA implementation possible.

No shared memory, tiling, or advanced optimizations yet — just one thread computing one output element of `C`.

We will measure that kernel first, identify its bottlenecks, and then optimize them one at a time.

👉 [Continue to the Naive CUDA GEMM experiment](./experiments/01_naive_cuda/)
