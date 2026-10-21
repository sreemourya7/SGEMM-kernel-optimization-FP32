# Kernel 1 — Naive CUDA GEMM

The CPU baseline gave us a useful starting point, but it also exposed the obvious next question: matrix multiplication contains a huge amount of independent work, so why are we asking a small number of CPU cores to process it mostly sequentially?

This is where CUDA enters the picture. The goal of this first GPU kernel is intentionally simple. I am not trying to optimize anything yet. I only want to take the same matrix multiplication we performed on the CPU and map it naturally onto the GPU.

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

The outer two loops enumerate every element of `C`. Instead of having the CPU execute those outer loops one element at a time, CUDA lets us create many threads and assign each thread one `(row, column)` position in `C`.

Something like this:

```text
Thread (0,0)  -> C[0][0]
Thread (0,1)  -> C[1][0]
Thread (0,2)  -> C[2][0]

Thread (1,0)  -> C[0][1]
Thread (1,1)  -> C[1][1]
Thread (1,2)  -> C[2][1]

...
```

Each thread still has to perform the dot product itself, but now many output elements(not all because we don't have as many threads as the number of entries to be computed in matrix C) can be computed at the same time.

> **[Insert Figure 1 here: CUDA Grid → Blocks → Threads]**

---

## How CUDA Organizes the Work

A CUDA kernel launch creates a **grid**. The grid contains multiple **thread blocks**, and each block contains multiple **threads**. For this kernel, I use a 2D thread block because the output matrix itself is two-dimensional.

For example:

```cpp
dim3 blockDim(32, 32);//defines a 2D thread block structure containing 1,024 total threads (32 threads in x and 32 threads in y)
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

For a 2D launch, the output position in this kernel is computed like this:

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

The block tells us which region of the matrix we are in, while the thread index tells us the position inside that block. Together they give the global `(x, y)` coordinate.

> **[Insert Figure 2 here: Thread → C[row][col] mapping]**

---

## What Does Each Thread Actually Do?

Once a thread knows which output element it owns, the computation is almost identical to the inner loop of the CPU implementation.

Conceptually:

```text
x = global thread row
y = global thread column

if x and y are inside C:

    sum = 0

    for k = 0 ... K-1:
        sum += A[x][k] * B[k][y]

    C[x][y] = sum
```

In CUDA-style pseudocode:

```cpp
__global__ void naive_gemm(A, B, C, M, N, K)
{
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N)
    {
        float sum = 0.0f;

        for (int i = 0; i < K; ++i)
        {
            sum += A[x * K + i]
                 * B[i * N + y];
        }

        C[x * N + y] = sum;
    }
}
```

The important difference from the CPU version is not the arithmetic itself. The important difference is **who performs it**.Instead of one CPU loop walking through the entire output matrix, thousands of GPU threads can each work on separate output elements.

---

## What Happens in Memory?

At first glance, this already looks much better. We have introduced massive parallelism without changing the underlying mathematics. But there is an important problem hiding inside the kernel.

Let's look at how thread's are accessing the memory in this kernel:

```text
For simplicity consider only thread(0, 0) and it's adjacent thread(0, 1), they compute C(0, 0) and C(1, 0) respectively
```

Their accesses:

```text
At i = 0
t(0, 0) access A(0, 0) and B(0, 0)
t(0, 1) access A(1, 0) and B(0, 0)
...
```

and:

```text
At i = 1
t(0, 0) access A(0, 1) and B(1, 0)
t(0, 1) access A(1, 1) and B(1, 0)
...
```
At an instance :
The accesses of `A` move across a column. Since the matrices are stored in row-major order, those elements at a distance of K columns(strided through memory).
The accesses to `B` are all the same. Let's look at the memory access pattern of Naive kernel in the cleanest way for best understanding of why this kernel is slow?

## Global Memory Access Pattern in the Naive Kernel

To understand why this first kernel leaves performance on the table, it helps to stop looking at a single thread and instead look at what an **entire warp** is doing at the same instruction.

Assume a `1024 × 1024` matrix and consider **Warp 0**, containing threads `t0 ... t31`. Under the thread mapping used in this naive kernel, these 32 threads compute different rows of `C` while sharing the same output column. During the dot-product loop, each thread accesses:

```cpp
A[row][i] * B[i][col]
```

where every thread in the warp executes the same value of `i` at the same time.

### Accesses to A

For matrix `A`, each thread accesses a different row but the same column:

| `i` | `t0` | `t1` | `t2` | `t3` | ... | `t31` |
|---:|---|---|---|---|---|---|
| `0` | `A[0][0]` | `A[1][0]` | `A[2][0]` | `A[3][0]` | ... | `A[31][0]` |
| `1` | `A[0][1]` | `A[1][1]` | `A[2][1]` | `A[3][1]` | ... | `A[31][1]` |
| `2` | `A[0][2]` | `A[1][2]` | `A[2][2]` | `A[3][2]` | ... | `A[31][2]` |
| `3` | `A[0][3]` | `A[1][3]` | `A[2][3]` | `A[3][3]` | ... | `A[31][3]` |
| `...` | `...` | `...` | `...` | `...` | ... | `...` |
| `1023` | `A[0][1023]` | `A[1][1023]` | `A[2][1023]` | `A[3][1023]` | ... | `A[31][1023]` |

The important thing is to read **across one row of this table**, because those are the memory requests issued by the warp together.

At `i = 0`:

```text
Thread:      t0          t1          t2          t3                 t31
             ↓           ↓           ↓           ↓                   ↓

Access:    A[0][0]     A[1][0]     A[2][0]     A[3][0]    ...    A[31][0]

Index:        0          1024         2048         3072      ...      31744
```

Since the matrix is stored in row-major order, adjacent threads are therefore separated by:

$$
1024\ \text{floats} \times 4\ \text{bytes/float}
= 4096\ \text{bytes}
$$

Instead of neighboring warp lanes requesting neighboring addresses, every lane jumps to a different row of `A`. This is a heavily **strided, non-coalesced access pattern**, so the warp cannot combine these requests into a small number of efficient contiguous memory transactions.

### Accesses to B

The situation for `B` is very different. Because all threads in this warp are computing the same output column, they request the **same element of B** at each iteration:

| `i` | `t0` | `t1` | `t2` | `t3` | ... | `t31` |
|---:|---|---|---|---|---|---|
| `0` | `B[0][0]` | `B[0][0]` | `B[0][0]` | `B[0][0]` | ... | `B[0][0]` |
| `1` | `B[1][0]` | `B[1][0]` | `B[1][0]` | `B[1][0]` | ... | `B[1][0]` |
| `2` | `B[2][0]` | `B[2][0]` | `B[2][0]` | `B[2][0]` | ... | `B[2][0]` |
| `...` | `...` | `...` | `...` | `...` | ... | `...` |
| `1023` | `B[1023][0]` | `B[1023][0]` | `B[1023][0]` | `B[1023][0]` | ... | `B[1023][0]` |

At a given iteration, such as `i = 0`, the warp effectively does this:

```text
t0 ──┐
t1 ──┤
t2 ──┤
t3 ──┤──→ B[0][0]
...  │
t31 ─┘
```

All 32 lanes request the same address, making this a **same-address / broadcast-friendly access** rather than 32 widely separated memory requests.

So, at the warp level, the naive kernel has a very asymmetric memory-access pattern:

```text
A : different rows, same column → large stride between lanes → non-coalesced ❌

B : same element requested by all lanes → broadcast-friendly             ✓
```

This immediately suggests our first optimization opportunity. The arithmetic itself does not need to change at all; instead, we can change the mapping between threads and output elements so that neighboring lanes access neighboring elements of `A`.

That is the idea behind the next kernel: **global memory coalescing**.


> **[Insert Figure 3 here: Naive global-memory access pattern for A and B]**

---

## Next Step

The first thing I want to investigate is the way neighboring threads access global memory.CUDA hardware is much happier when threads in a warp request nearby memory addresses that can be serviced efficiently together. Our current thread mapping does not necessarily make the best use of that behavior. So the next kernel will keep the exact same GEMM computation while changing how threads map onto matrix elements.
Our next kernel optimization is this:

> **Make global-memory access more coalesced and measure what changes.**

---

### Implementation and Results

The full CUDA implementation is available here:

[View `naive_gemm.cu`](./naive_gemm.cu)

Benchmark configuration, correctness checks, raw measurements, and final performance results are recorded here:

[View benchmark results](./results.md)
