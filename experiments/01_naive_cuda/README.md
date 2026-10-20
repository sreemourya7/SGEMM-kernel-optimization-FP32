# Kernel 1 — Naive CUDA GEMM

The CPU baseline gave us a useful starting point, but it also exposed the obvious next question: matrix multiplication contains a huge amount of independent work, so why are we asking a small number of CPU cores to process it mostly sequentially?

This is where CUDA enters the picture.

The goal of this first GPU kernel is intentionally simple. I am not trying to optimize anything yet. I only want to take the same matrix multiplication we performed on the CPU and map it naturally onto the GPU.

The basic idea is:

> **one CUDA thread computes one element of the output matrix `C`.**

That gives us the simplest possible GPU implementation against which every future optimization can be compared.

---

## From a CPU Loop to Thousands of GPU Threads

On the CPU, matrix multiplication looked roughly like this:

```text
for every row i of A
    for every column j of B
        sum = 0

        for k = 0 ... K-1
            sum += A[i][k] * B[k][j]

        C[i][j] = sum
```

The outer two loops enumerate every element of `C`.

Instead of having the CPU execute those outer loops one element at a time, CUDA lets us create many threads and assign each thread one `(row, column)` position in `C`.

Conceptually:

```text
Thread (0,0)  -> C[0][0]
Thread (0,1)  -> C[0][1]
Thread (0,2)  -> C[0][2]

Thread (1,0)  -> C[1][0]
Thread (1,1)  -> C[1][1]
Thread (1,2)  -> C[1][2]

...
```

Each thread still has to perform the dot product itself, but now many output elements can be computed at the same time.

> **[Insert Figure 1 here: CUDA Grid → Blocks → Threads]**

---

## How CUDA Organizes the Work

A CUDA kernel launch creates a **grid**. The grid contains multiple **thread blocks**, and each block contains multiple **threads**.

For this kernel, I use a two-dimensional thread block because the output matrix itself is two-dimensional.

For example:

```cpp
dim3 blockDim(32, 32);
```

This creates:

```text
32 × 32 = 1024 threads per block
```

Each block covers a rectangular region of the output matrix.

The grid then contains enough blocks to cover the entire matrix:

```cpp
dim3 gridDim(
    (N + blockDim.x - 1) / blockDim.x,
    (M + blockDim.y - 1) / blockDim.y
);
```

The slightly unusual-looking division is just ceiling division. If the matrix dimensions are not exact multiples of the block dimensions, we still launch enough blocks to cover the remaining elements.

---

## Which Element Does a Thread Compute?

Every CUDA thread can determine its own position using CUDA's built-in variables:

```cpp
blockIdx
blockDim
threadIdx
```

For a two-dimensional launch, the output position can be computed as:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

The block tells us which region of the matrix we are in, while the thread index tells us the position inside that block.

Together they give the global `(row, col)` coordinate.

> **[Insert Figure 2 here: Thread → C[row][col] mapping]**

For example, suppose a thread determines:

```text
row = 5
col = 7
```

That thread becomes responsible for exactly one value:

```text
C[5][7]
```

It now needs row `5` from `A` and column `7` from `B`.

---

## What Does Each Thread Actually Do?

Once a thread knows which output element it owns, the computation is almost identical to the inner loop of the CPU implementation.

Conceptually:

```text
row = global thread row
col = global thread column

if row and col are inside C:

    sum = 0

    for k = 0 ... K-1:
        sum += A[row][k] * B[k][col]

    C[row][col] = sum
```

In CUDA-style pseudocode:

```cpp
__global__ void naive_gemm(A, B, C, M, N, K)
{
    row = blockIdx.y * blockDim.y + threadIdx.y;
    col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k]
                 * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}
```

The important difference from the CPU version is not the arithmetic itself.

The important difference is **who performs it**.

Instead of one CPU loop walking through the entire output matrix, thousands of GPU threads can each work on separate output elements.

---

## What Happens in Memory?

At first glance, this already looks much better. We have introduced massive parallelism without changing the underlying mathematics.

But there is an important problem hiding inside the kernel.

Consider what one thread reads while computing:

```text
C[row][col]
```

It accesses:

```text
A[row][0]
A[row][1]
A[row][2]
...
```

and:

```text
B[0][col]
B[1][col]
B[2][col]
...
```

The accesses to `A` move across a row. Since the matrices are stored in row-major order, those elements are adjacent in memory.

The accesses to `B`, however, move down a column. Consecutive elements used by one thread are separated by an entire row of `B`.

Conceptually:

```text
A:

A[row][0] -> A[row][1] -> A[row][2] -> A[row][3]
    contiguous in memory
```

while:

```text
B:

B[0][col]
     |
B[1][col]
     |
B[2][col]
     |
B[3][col]

strided through memory
```

> **[Insert Figure 3 here: Naive global-memory access pattern for A and B]**

There is another issue as well. Neighboring threads often need overlapping data from `A` and `B`, yet in this naive implementation every thread independently fetches the values it needs from global memory.

For example, multiple threads computing values in the same row of `C` will repeatedly use the same row of `A`.

Likewise, multiple threads computing values in the same column of `C` will repeatedly use the same column of `B`.

The kernel does not explicitly reuse any of that data.

It simply asks global memory for it again.

---

## Why Call This Kernel "Naive"?

This kernel is naive not because the mathematics is wrong, but because we have done almost nothing to help the GPU use its hardware efficiently.

We have exposed parallelism, which is already a major step beyond the CPU version, but the kernel still performs a large number of global-memory accesses and makes no attempt to reuse data between threads.

There is no shared-memory tiling.

There is no register tiling.

There is no deliberate reuse of values loaded from global memory.

There is no optimization of the memory-access pattern yet.

This kernel therefore gives us something extremely valuable:

> **a clean GPU baseline.**

From here on, every optimization can answer a concrete question:

> What hardware inefficiency are we fixing, and how much performance does fixing it buy us?

---

## Next Step

The first thing I want to investigate is the way neighboring threads access global memory.

CUDA hardware is much happier when threads in a warp request nearby memory addresses that can be serviced efficiently together.

Our current thread mapping does not necessarily make the best use of that behavior.

So the next kernel will keep the exact same GEMM computation while changing how threads map onto matrix elements.

The goal will be simple:

> **Make global-memory access more coalesced and measure what changes.**

---

### Implementation and Results

The full CUDA implementation is available here:

[View `naive_gemm.cu`](./naive_gemm.cu)

Benchmark configuration, correctness checks, raw measurements, and final performance results are recorded here:

[View benchmark results](./results.md)
