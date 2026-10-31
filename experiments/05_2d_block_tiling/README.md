# Kernel 05 — 2D Block Tiling + Autotuning | FP32 SGEMM

## 1. Motivation — Can We Reuse Both A and B?

In [Kernel 04 — 1D Block Tiling](../04_1d_block_tiling/), we improved GEMM performance by changing the amount of work assigned to each thread.

Instead of computing a single output element of C, each thread computed **eight vertically adjacent output elements**, maintaining eight accumulators in registers.

For each reduction step along K, the thread:

- Loaded eight A values from shared memory.
- Loaded one B value from shared memory.
- Reused that B value across eight fused multiply-add (FMA) operations.

This increased the amount of computation performed per shared-memory load and significantly reduced the MIO Throttle observed in our earlier kernel.

However, one limitation remained.

**We were reusing B across multiple FMAs, but each A value was still used only once within the thread's computation for that reduction step.**

This raised a natural question:

> If reusing one B value across multiple A values helped so much, what if we could reuse both A and B within each thread?

That became the motivation for **2D Block Tiling**.

---

## 2. The Core Idea — One Thread Computes a 2D Patch of C

Instead of assigning a vertical strip of output elements to each thread, we assign a small rectangular patch.

Suppose a thread computes an 8×8 patch of C.

For one reduction index `k`, it needs:

- Eight A values corresponding to eight output rows.
- Eight B values corresponding to eight output columns.

These values can be held temporarily in registers.

```text
                 B register fragment
                  b0  b1  b2 ... b7
                ┌───────────────────┐
             a0 │ •   •   •  ... •  │
             a1 │ •   •   •  ... •  │
             a2 │ •   •   •  ... •  │
A registers  a3 │ •   •   •  ... •  │
             a4 │ •   •   •  ... •  │
             a5 │ •   •   •  ... •  │
             a6 │ •   •   •  ... •  │
             a7 │ •   •   •  ... •  │
                └───────────────────┘
                    C: 8 × 8
```

Each A value participates in eight FMAs, while each B value participates in eight FMAs.

Therefore:

**8 A values + 8 B values → 64 FMAs.**

This is an *outer-product computation*.

Rather than repeatedly fetching operands for individual output elements, the thread loads small fragments and uses every combination of A and B values.

The corresponding CUDA logic is:

```cpp
float acc[TM][TN] = {};
float regA[TM];
float regB[TN];

for (int k = 0; k < BK; ++k) {
    // Load A and B fragments from shared memory
    for (int i = 0; i < TM; ++i)
        regA[i] = As[threadRow * TM + i][k];

    for (int j = 0; j < TN; ++j)
        regB[j] = Bs[k][threadCol * TN + j];

    // Outer-product accumulation
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = fmaf(regA[i], regB[j], acc[i][j]);
        }
    }
}
```

Here, `TM` and `TN` determine how many output rows and columns belong to one thread.

The essential improvement is that **we perform more arithmetic using each group of shared-memory operands**.

This should reduce shared-memory instruction demand relative to useful computation, potentially relieving MIO pipeline pressure.

It also provides more arithmetic that may help tolerate some memory dependencies, although reducing Long Scoreboard stalls is not guaranteed.

---

## 3. Our Initial Thought — Why Not Use 8×8 Per Thread?

A natural starting point was:

```cpp
BM = 128;
BN = 128;
BK = 8;

TM = 8;
TN = 8;
```

The corresponding matrix tiles are:

```text
A: 128 × 8
     ×
B:   8 × 128
     =
C: 128 × 128
```

Each thread computes:

`TM × TN = 8 × 8 = 64` output elements.

Since the output tile contains `128 × 128 = 16,384` elements, we require:

`16,384 / 64 = 256 threads per block`

That gives us eight warps per block.

At first glance, this looks very attractive:

- Each thread performs 64 FMAs for every inner K step.
- Both A and B are reused across multiple output elements.
- The larger output tile allows more computation from each pair of shared-memory tiles.
- The number of threads per block remains manageable.

But there is an important tradeoff.

### Register pressure

An 8×8 output patch requires **64 FP32 accumulator values per thread**.

That is before accounting for A/B fragments, addresses, indexing, and temporary variables.

On our GTX 1650, registers are a limited resource shared by all resident threads on an SM.

Increasing the work assigned to each thread can improve reuse, but excessive register allocation can reduce the number of active warps or introduce spilling.

So we faced an interesting question:

> Is maximizing per-thread data reuse always worth the additional register pressure?

Rather than assuming 8×8 would be optimal, we decided to experiment.

---

## 4. Start Small — Establishing a 4×4 Baseline

We began with a more conservative configuration:

```cpp
BM = 64;
BN = 64;
BK = 8;

TM = 4;
TN = 4;
```

This gives us:

```text
A: 64 × 8
     ×
B:  8 × 64
     =
C: 64 × 64
```

Each thread computes a 4×4 patch, maintaining 16 output accumulators in registers.

The thread count is:

`(64 / 4) × (64 / 4) = 256 threads per block`

For each reduction step, the thread obtains four A values and four B values and performs:

**4 × 4 = 16 FMAs.**

Why begin here?

We wanted to validate the 2D thread-to-output mapping, cooperative global-memory loading, shared-memory layout, register accumulation, and correctness before increasing the tile sizes.

This also gave us a baseline from which we could change individual tuning parameters and understand their effects.

Once the baseline worked, we turned to the first tuning question.

---

## 5. Autotuning Stage I — How Deep Should the K-Tile Be?

Our first experiment kept the output tile and per-thread computation fixed:

```cpp
BM = 64;
BN = 64;

TM = 4;
TN = 4;
```

We varied only `BK`:

| Experiment | A tile | B tile | BK |
|---|---|---|---|
| A | 64×8 | 8×64 | 8 |
| B | 64×16 | 16×64 | 16 |
| C | 64×32 | 32×64 | 32 |

### Why vary BK?

GEMM proceeds through the K dimension in tiles.

For every K-tile, the block must:

1. Load the corresponding A and B tiles from global memory.
2. Synchronize the threads.
3. Compute using those shared-memory tiles.
4. Synchronize before overwriting the shared-memory buffers.

A small BK means the block repeats this process more frequently.

Increasing BK lets the block perform more arithmetic between successive tile-loading phases, potentially reducing the relative cost of synchronization and loop control.

But increasing BK also enlarges the shared-memory buffers and may increase register usage or generated code size.

There is another subtle point:

**Increasing BK alone does not improve the ideal global-memory arithmetic intensity when BM and BN remain fixed.** The block performs proportionally more computation but also loads proportionally more input data.

We therefore wanted to identify a useful balance rather than simply choose the largest BK.

After exploring K-tile depth, the next question became more interesting.

---

## 6. Autotuning Stage II — Does the Output Tile Need to Be Square?

Our initial output tile was 64×64.

But matrix multiplication does not require the output tile to be square.

An A tile of size `BM × BK` and a B tile of size `BK × BN` produce a `BM × BN` output tile.

Only the inner BK dimension needs to match.

So why restrict ourselves to equal BM and BN?

We expanded our search to include rectangular tiles and different per-thread output shapes.

| Configuration | A tile | B tile | Outputs/thread |
|---|---|---|---|
| 8×4 | 128×16 | 16×64 | 8×4 |
| 4×8 | 64×16 | 16×128 | 4×8 |
| 8×8 | 128×8 | 8×128 | 8×8 |
| 8×8 | 128×16 | 16×128 | 8×8 |

All configurations maintained 256 threads per block.

### Why compare 8×4 with 4×8?

Both configurations compute **32 outputs per thread**.

They also perform the same number of FMAs per reduction step and have equivalent conceptual operand reuse:

- 8×4: eight A values and four B values.
- 4×8: four A values and eight B values.

Both use twelve scalar operands to perform 32 FMAs.

However, this does not mean the GPU executes them identically.

The two shapes change how threads map onto rows and columns, which can influence:

- Global-memory access and coalescing.
- Shared-memory access patterns.
- Compiler instruction selection and vectorization.
- Register allocation and instruction scheduling.

This was an important realization:

> Two mathematically equivalent tiling strategies can generate very different GPU execution behavior.

### What about 8×8?

We also tested the original, more aggressive idea.

An 8×8 output patch offers greater reuse within each thread, but it requires substantially more live accumulator values.

The extra reuse must compensate for the register resources consumed.

In our experiments, the rectangular **8×4 layout emerged as the stronger choice**.

This challenged our original assumption that a larger, symmetric register tile would necessarily be better.

Instead, our search pointed toward an asymmetric output tile.

---

## 7. Autotuning Stage III — Refining the Winning 8×4 Shape

After selecting the promising 8×4 layout, we fixed:

```cpp
BM = 128;
BN = 64;

TM = 8;
TN = 4;

THREADS = 256;
```

Then we revisited the K-tile depth.

| Experiment | A tile | B tile | BK |
|---|---|---|---|
| A | 128×8 | 8×64 | 8 |
| B | 128×16 | 16×64 | 16 |
| C | 128×32 | 32×64 | 32 |

This was our final focused sweep.

We already knew the output shape was promising. Now we wanted to determine how much K-dimension work each block should process before refreshing its shared-memory tiles.

The intermediate BK=16 configuration provided the best balance in our tests.

With BK=8, the kernel needed more frequent shared-memory loading and synchronization phases.

With BK=32, the larger tile increased resource requirements, which could reduce available parallelism despite the additional work per tile.

The profiler would be needed to establish the precise hardware contribution of each factor, but our tuning experiments gave us a clear configuration to select.

We therefore finalized:

```cpp
constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 16;

constexpr int TM = 8;
constexpr int TN = 4;

constexpr int THREADS = 256;
```

The important conclusion was not simply that BK=16 worked well.

It was that **tile sizes must be selected together**, considering reuse, memory access patterns, register pressure, synchronization, and available parallelism.

---

## 8. How Our Final 2D Block-Tiled Kernel Works

Now that we've established why we selected the configuration, let's walk through one thread block's execution.

### Step 1 — Assign a 128×64 output tile

Each thread block computes one 128×64 region of C.

```text
A tile              B tile                C tile

128 × 16     ×      16 × 64       →        128 × 64
```

A block contains 256 threads.

Each thread owns an 8×4 output patch, so the entire output tile is covered by:

- 16 groups of threads along the row dimension.
- 16 groups of threads along the column dimension.

That is `16 × 16 = 256` threads.

Each thread maintains **32 independent FP32 accumulators** throughout the GEMM reduction.

### Step 2 — Cooperatively load A and B into shared memory

For one K-tile:

- A contains `128 × 16 = 2,048` elements.
- B contains `16 × 64 = 1,024` elements.

With 256 cooperating threads, each thread is responsible for:

- Eight scalar A elements.
- Four B elements.

The optimized layout uses padding for the A shared-memory tile and aligned `float4` access for B where applicable.

The threads collectively populate the shared-memory buffers and synchronize before computation begins.

### Step 3 — Load register fragments and compute an outer product

For each of the 16 reduction steps within the K-tile, a thread reads:

- Eight A values from shared memory into registers.
- Four B values from shared memory into registers.

The thread computes their outer product.

```text
          Four B values
           b0  b1  b2  b3
         ┌────────────────┐
      a0 │ •   •   •   •  │
      a1 │ •   •   •   •  │
      a2 │ •   •   •   •  │
      a3 │ •   •   •   •  │
      a4 │ •   •   •   •  │
      a5 │ •   •   •   •  │
      a6 │ •   •   •   •  │
      a7 │ •   •   •   •  │
         └────────────────┘
            32 outputs
```

For every K step:

**8 A values × 4 B values = 32 FMAs.**

Each A value is reused across four columns, and each B value is reused across eight rows.

The results accumulate directly into the thread's 32 register-resident output values.

No intermediate C values need to be written to global memory.

### Step 4 — Move to the next K-tile

After processing all 16 reduction positions, the block synchronizes before reusing the shared-memory buffers.

It advances by BK along the K dimension and repeats the same loading and accumulation process.

The thread retains its output accumulators across all K-tiles.

### Step 5 — Write the final C values

Once the complete K dimension has been processed, each thread writes its 8×4 output patch to global memory.

The implementation uses vectorized output stores when the addresses and tile boundaries permit them.

The central data flow is:

```text
        GLOBAL MEMORY
          A      B
          |      |
          v      v
      SHARED MEMORY
      A tile   B tile
          \    /
           \  /
        REGISTER FRAGMENTS
         8 A × 4 B
              |
         32 FMAs per k
              |
      REGISTER ACCUMULATORS
        8 × 4 outputs
              |
       Repeat across K
              |
        GLOBAL MEMORY
           C tile
```

---

## 9. What Did This Optimization Teach Us?

Our journey began with a simple idea:

**If one-dimensional register tiling improved reuse, two-dimensional register tiling should let us reuse both operands.**

But implementing 2D tiling raised a second question: how much work should each thread perform?

Increasing the output patch improves potential operand reuse, but it also requires more registers. Enlarging the K-tile can reduce synchronization overhead, but also consumes more resources. Changing the output tile shape affects how threads access and distribute data.

Our experiments showed that these decisions interact.

The final configuration was not our initial 8×8 idea, nor the largest tile we tested.

It was an **8×4 per-thread patch within a 128×64×16 block tile**.

That is the central lesson of Kernel 05:

> **Effective GEMM optimization is not about maximizing one resource or one kind of reuse. It is about balancing data reuse, instruction demand, memory access, and execution parallelism.**

Our next step is to use Nsight Compute to investigate whether these changes reduced MIO Throttle and Long Scoreboard pressure as hypothesized.

Detailed benchmark comparisons belong in [`results.md`](results.md), while the hardware-level investigation will be documented separately in [`profiling.md`](profiling.md).

---