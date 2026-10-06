# Experiment 00 — Why GPUs for Matrix Multiplication?

Before writing a single CUDA kernel, I wanted to understand the problem we are trying to accelerate.

Why does matrix multiplication matter so much?  
How expensive is it on a straightforward CPU implementation?  
And what makes GPUs particularly good at this kind of computation?

Let's start from the beginning.

---

## 1. What is GEMM?

**GEMM** stands for **General Matrix Multiplication**.

The general operation is:

\[
C = α*AB + β*C
\]

where `A`, `B`, and `C` are matrices and `α` and `β` are scalar values.

For now, lets focus on the simpler case:

\[
C = AB
\]

Before understanding how to optimize GEMM, lets understand where exactly is Matrix multiplication used and why it is one of the fundamental computational building blocks behind modern machine learning.

Operations inside:

- linear layers,
- attention,
- MLPs,
- projections,
- convolutions after suitable transformations,

these eventually involve enormous amounts of matrix multiplication.

That makes GEMM an excellent workload for learning GPU performance engineering.

If we understand how to make GEMM fast, we start understanding many of the same ideas used to make modern AI systems fast.

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

**one row from A**

and computing its dot product with:

**one column from B**

For example:

```text
c00 = a00*b00
    + a01*b10
    + a02*b20
```

In general:

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik}B_{kj}
$$

The interesting part is that `C[0][0]`, `C[0][1]`, `C[1][0]` and so on can largely be computed independently.

It's worth keeping this thought in mind.

It is going to become very important when we move to the GPU.

---

## 3. Why does GEMM require approximately `2N³` FLOPs?

For two square `N × N` matrices, the result contains:

\[
N^2
\]

elements.

To compute one output element, we perform approximately `N` iterations of:

```cpp
sum += A[i * N + k] * B[k * N + j];
```

Each iteration contains:

- one floating-point multiplication
- one floating-point addition

So we count approximately:

\[
2N
\]

floating-point operations per output element.

Since there are `N²` output elements:

\[
N^2 \times 2N = 2N^3
\]

Therefore:

\[
\boxed{\text{FLOPs} \approx 2N^3}
\]

For example, with:

```text
N = 4096
```

we perform approximately:

\[
2(4096)^3 \approx 137.4 \text{ billion FLOPs}
\]

Three innocent-looking loops have suddenly turned into **137 billion operations**.

Things escalated quickly.

---

## 4. Starting with the simplest CPU implementation

Before touching CUDA, I wanted a baseline.

The most direct implementation of matrix multiplication follows the mathematical definition almost exactly:

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

Nothing fancy.

- `i` chooses a row of `A`
- `j` chooses a column of `B`
- `k` performs the dot product

This is intentionally a **naive CPU implementation**.

It is **not** intended to represent the performance of optimized CPU libraries such as Intel MKL or OpenBLAS.

The goal is simply to establish a clean starting point that we can measure and later compare against CUDA.

👉 [View the full CPU implementation](./experiments/00_cpu_baseline/cpu_gemm.cpp)

---

## 5. How fast is my CPU implementation?

Runtime alone is not enough.

A 2-second runtime means very different things depending on whether we performed one million or one trillion operations.

Since we know approximately how much work GEMM performs, we can calculate computational throughput:

\[
GFLOP/s =
\frac{2N^3}
{\text{execution time} \times 10^9}
\]

I will benchmark multiple matrix sizes and measure both execution time and achieved GFLOP/s.

| Matrix Size | Execution Time | Performance |
|---:|---:|---:|
| `512 × 512` | TBD | TBD GFLOP/s |
| `1024 × 1024` | TBD | TBD GFLOP/s |
| `2048 × 2048` | TBD | TBD GFLOP/s |
| `4096 × 4096` | TBD | TBD GFLOP/s |

These values will be replaced with actual measurements from my machine.

👉 [View detailed benchmark results](./experiments/00_cpu_baseline/results.md)

The important number for this project is going to be:

> **How many GFLOP/s did my implementation actually achieve?**

Because soon we will compare that against what the GPU hardware is theoretically capable of.

---

## 6. Now compare that with my GTX 1650

My GPU is an **NVIDIA GeForce GTX 1650**.

Depending on the exact GTX 1650 variant and clock speed, its theoretical FP32 throughput is roughly:

\[
\approx 3 \text{ TFLOP/s}
\]

or approximately:

\[
3000 \text{ GFLOP/s}
\]

The rough idea behind peak FP32 throughput is:

\[
\text{CUDA cores}
\times
\text{clock frequency}
\times
2
\]

The `×2` comes from a fused multiply-add operation:

```text
a * b + c
```

which performs one multiplication and one addition.

But there is a huge catch.

**Theoretical performance is not the same as achieved performance.**

Owning a GPU capable of ~3 TFLOP/s does not mean every CUDA program automatically runs at ~3 TFLOP/s.

A poorly designed kernel can leave a massive amount of that hardware sitting idle.

And this difference between:

```text
The performance the hardware CAN provide
```

and

```text
The performance our kernel ACTUALLY achieves
```

is where GPU performance engineering gets interesting.

The goal of this project is to gradually close that gap.

---

## 7. Why is GEMM so well suited to GPUs?

Look at the output matrix again:

```text
C[0][0]   C[0][1]   C[0][2]   ...
C[1][0]   C[1][1]   C[1][2]   ...
C[2][0]   C[2][1]   C[2][2]   ...
...
```

Each of these output elements requires essentially the same kind of computation.

And many of them can be computed independently.

That means matrix multiplication exposes an enormous amount of **data parallelism**.

CPUs are designed around a relatively small number of powerful cores optimized for things such as:

- complex control flow,
- branch-heavy workloads,
- low-latency execution,
- operating systems,
- general-purpose applications.

GPUs take a different approach.

They devote far more hardware toward executing huge numbers of similar arithmetic operations concurrently.

And GEMM happens to contain millions — sometimes billions — of operations that look like:

```text
multiply
add
multiply
add
multiply
add
...
```

Matrix multiplication basically walks up to the GPU and says:

> I have several billion multiply-add operations that can run in parallel.

And the GPU says:

> Finally, someone understands me.

This is one reason GPUs became so important for modern deep learning.

If we tried to build today's AI systems around naive CPU matrix multiplication alone, we might finish training the model just in time for our grandchildren to benchmark it.

---

## 8. Isn't there a mathematically faster matrix multiplication algorithm?

The classical matrix multiplication algorithm used above has complexity:

\[
O(N^3)
\]

There are algorithms with better asymptotic complexity.

For example, **Strassen's algorithm** reduces the complexity to approximately:

\[
O(N^{2.807})
\]

and later algorithms improved the theoretical exponent even further.

So why don't high-performance GPU libraries simply use the algorithm with the smallest Big-O complexity everywhere?

Because Big-O notation is only part of the performance story.

Real hardware cares about much more than the number of arithmetic operations.

Performance can depend heavily on:

- memory access patterns,
- cache behavior,
- memory bandwidth,
- data reuse,
- synchronization,
- register pressure,
- occupancy,
- instruction throughput,
- parallelism,
- and hardware utilization.

An algorithm can perform fewer operations on paper while still running slower for practical problem sizes because of larger constants or poor interaction with the hardware.

This leads to one of the most important ideas I want to explore throughout this project:

> **The fastest algorithm on paper is not necessarily the fastest implementation on real hardware.**

GPU performance engineering is about understanding that gap.

Not just asking:

> How many operations does this algorithm perform?

but also:

> **How efficiently can the hardware execute those operations?**

And that is what the next experiments will investigate.

---

## Next: Naive CUDA GEMM

We now have:

- a mathematical understanding of GEMM,
- a simple CPU implementation,
- a way to measure performance,
- and a rough idea of what the GPU hardware is capable of.

The next step is straightforward:

> **Take the exact same computation and move it onto the GPU.**

No shared memory.

No fancy tiling.

No clever optimization.

Just the simplest CUDA GEMM kernel possible.

Then we measure it.

And from there, we optimize one bottleneck at a time.
