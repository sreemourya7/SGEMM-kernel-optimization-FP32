
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Kernel 07: 2D tiled SGEMM with register prefetching.
// Target: NVIDIA GTX 1650, sm_75.
// Row-major C = alpha * A * B + beta * C.

#ifndef BM_SIZE
#define BM_SIZE 128
#endif

#ifndef BN_SIZE
#define BN_SIZE 64
#endif

#ifndef BK_SIZE
#define BK_SIZE 16
#endif

#ifndef TM_SIZE
#define TM_SIZE 8
#endif

#ifndef TN_SIZE
#define TN_SIZE 4
#endif

constexpr int BM = BM_SIZE;
constexpr int BN = BN_SIZE;
constexpr int BK = BK_SIZE;
constexpr int TM = TM_SIZE;
constexpr int TN = TN_SIZE;

constexpr int THREADS = (BM / TM) * (BN / TN);
constexpr int COL_GROUPS = BN / TN;
constexpr int B_COL_VECTORS = BN / 4;
constexpr int B_VECTORS = BK * B_COL_VECTORS;
constexpr int A_LOADS = (BM * BK) / THREADS;

static_assert(TM == 4 || TM == 8, "TM must be 4 or 8");
static_assert(TN == 4 || TN == 8, "TN must be 4 or 8");
static_assert(BM % TM == 0 && BN % TN == 0,
              "Invalid thread tile");
static_assert((BM * BK) % THREADS == 0,
              "Invalid A loading distribution");
static_assert(THREADS == 256,
              "Expected 256 threads per block");

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHECK(call)                                     \
    do                                                       \
    {                                                        \
        cudaError_t err = (call);                            \
        if (err != cudaSuccess)                              \
        {                                                    \
            std::fprintf(stderr,                             \
                         "CUDA error at line %d: %s\n",      \
                         __LINE__, cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE);                         \
        }                                                    \
    } while (0)

// ============================================================
// KERNEL 07: REGISTER PREFETCHING
// ============================================================

template <bool FULL>
__global__ void sgemm_block2d_prefetch(
    int M, int N, int K,
    float alpha,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float beta,
    float *__restrict__ C)
{
    // Preserve the shared-memory layouts from Kernel 05.
    __shared__ float As[BM][BK + 1];
    __shared__ float4 Bs[BK][B_COL_VECTORS];

    const int tid = threadIdx.x;

    const int threadRow = tid / COL_GROUPS;
    const int threadCol = tid % COL_GROUPS;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int outRow = blockRow + threadRow * TM;
    const int outCol = blockCol + threadCol * TN;

    // Each thread maintains TM x TN output accumulators.
    float acc[TM][TN] = {};

    // ========================================================
    // LOOP OVER K-TILES
    // ========================================================

    for (int bk = 0; bk < K; bk += BK)
    {

        // ----------------------------------------------------
        // STEP 1: GLOBAL A -> SHARED A
        // Unchanged from Kernel 05.
        // ----------------------------------------------------

#pragma unroll
        for (int load = 0; load < A_LOADS; ++load)
        {

            const int idx = tid + load * THREADS;

            const int aRow = idx / BK;
            const int aK = idx % BK;

            const int globalRow = blockRow + aRow;
            const int globalK = bk + aK;

            if constexpr (FULL)
            {

                As[aRow][aK] =
                    A[static_cast<size_t>(globalRow) * K + globalK];
            }
            else
            {

                As[aRow][aK] =
                    (globalRow < M && globalK < K)
                        ? A[static_cast<size_t>(globalRow) * K + globalK]
                        : 0.0f;
            }
        }

        // ----------------------------------------------------
        // STEP 2: GLOBAL B -> SHARED B
        // Existing float4 vectorization.
        // ----------------------------------------------------

#pragma unroll
        for (int vec = tid;
             vec < B_VECTORS;
             vec += THREADS)
        {
            const int bK = vec / B_COL_VECTORS;
            const int bColGroup = vec % B_COL_VECTORS;

            const int globalK = bk + bK;
            const int globalCol =
                blockCol + bColGroup * 4;

            const size_t offset =
                static_cast<size_t>(globalK) * N + globalCol;

            if constexpr (FULL)
            {

                Bs[bK][bColGroup] =
                    *reinterpret_cast<const float4 *>(
                        B + offset);
            }
            else
            {

                float4 v =
                    make_float4(0.f, 0.f, 0.f, 0.f);

                if (globalK < K)
                {

                    if (globalCol + 0 < N)
                        v.x = B[offset + 0];

                    if (globalCol + 1 < N)
                        v.y = B[offset + 1];

                    if (globalCol + 2 < N)
                        v.z = B[offset + 2];

                    if (globalCol + 3 < N)
                        v.w = B[offset + 3];
                }

                Bs[bK][bColGroup] = v;
            }
        }

        __syncthreads();

        // ====================================================
        // STEP 3: REGISTER PREFETCHING
        //
        // Current step k:
        //   - Compute using current register fragments.
        //
        // Next step k+1:
        //   - Load A/B from shared memory into registers
        //     before executing the current step's FMAs.
        //
        // This is register prefetching, not cp.async.
        // ====================================================

        const int bBase = threadCol * (TN / 4);

        float aCur[TM];
        float aNext[TM];

        float4 bCur0 = Bs[0][bBase];
        float4 bNext0 =
            make_float4(0.f, 0.f, 0.f, 0.f);

        float4 bCur1 =
            make_float4(0.f, 0.f, 0.f, 0.f);

        float4 bNext1 =
            make_float4(0.f, 0.f, 0.f, 0.f);

        if constexpr (TN == 8)
        {
            bCur1 = Bs[0][bBase + 1];
        }

// Prime the pipeline: preload K-step 0.
#pragma unroll
        for (int i = 0; i < TM; ++i)
        {
            aCur[i] =
                As[threadRow * TM + i][0];
        }

        // ----------------------------------------------------
        // SOFTWARE-PIPELINED INNER K LOOP
        // ----------------------------------------------------

#pragma unroll
        for (int k = 0; k < BK; ++k)
        {

            // -----------------------------------------------
            // A. Prefetch operands for the NEXT K-step.
            // -----------------------------------------------

            if (k + 1 < BK)
            {

                bNext0 = Bs[k + 1][bBase];

                if constexpr (TN == 8)
                {
                    bNext1 =
                        Bs[k + 1][bBase + 1];
                }

#pragma unroll
                for (int i = 0; i < TM; ++i)
                {

                    aNext[i] =
                        As[threadRow * TM + i][k + 1];
                }
            }

            // -----------------------------------------------
            // B. Compute using CURRENT register operands.
            // -----------------------------------------------

#pragma unroll
            for (int i = 0; i < TM; ++i)
            {

                const float a = aCur[i];

                acc[i][0] =
                    fmaf(a, bCur0.x, acc[i][0]);

                acc[i][1] =
                    fmaf(a, bCur0.y, acc[i][1]);

                acc[i][2] =
                    fmaf(a, bCur0.z, acc[i][2]);

                acc[i][3] =
                    fmaf(a, bCur0.w, acc[i][3]);

                if constexpr (TN == 8)
                {

                    acc[i][4] =
                        fmaf(a, bCur1.x, acc[i][4]);

                    acc[i][5] =
                        fmaf(a, bCur1.y, acc[i][5]);

                    acc[i][6] =
                        fmaf(a, bCur1.z, acc[i][6]);

                    acc[i][7] =
                        fmaf(a, bCur1.w, acc[i][7]);
                }
            }

            // -----------------------------------------------
            // C. Advance register fragments.
            // -----------------------------------------------

            if (k + 1 < BK)
            {

                bCur0 = bNext0;

                if constexpr (TN == 8)
                {
                    bCur1 = bNext1;
                }

#pragma unroll
                for (int i = 0; i < TM; ++i)
                {
                    aCur[i] = aNext[i];
                }
            }
        }

        // Protect shared-memory tiles before overwrite.
        __syncthreads();
    }

    // ========================================================
    // STEP 4: WRITE REGISTER ACCUMULATORS TO GLOBAL C
    // Preserve Kernel 05's vectorized stores.
    // ========================================================

#pragma unroll
    for (int i = 0; i < TM; ++i)
    {

        const int row = outRow + i;

        if constexpr (FULL)
        {

#pragma unroll
            for (int v = 0; v < TN / 4; ++v)
            {

                const int j = 4 * v;

                const size_t offset =
                    static_cast<size_t>(row) * N + outCol + j;

                float4 out = make_float4(
                    alpha * acc[i][j],
                    alpha * acc[i][j + 1],
                    alpha * acc[i][j + 2],
                    alpha * acc[i][j + 3]);

                float4 *dst =
                    reinterpret_cast<float4 *>(
                        C + offset);

                if (beta != 0.0f)
                {

                    const float4 old = *dst;

                    out.x += beta * old.x;
                    out.y += beta * old.y;
                    out.z += beta * old.z;
                    out.w += beta * old.w;
                }

                *dst = out;
            }
        }
        else if (row < M)
        {

#pragma unroll
            for (int j = 0; j < TN; ++j)
            {

                const int col = outCol + j;

                if (col < N)
                {

                    const size_t idx =
                        static_cast<size_t>(row) * N + col;

                    const float value =
                        alpha * acc[i][j];

                    C[idx] = (beta == 0.0f)
                                 ? value
                                 : value + beta * C[idx];
                }
            }
        }
    }
}

// ============================================================
// KERNEL LAUNCH
// ============================================================

static void launch_sgemm(
    int N,
    const float *A,
    const float *B,
    float *C)
{
    const dim3 grid(
        CEIL_DIV(N, BN),
        CEIL_DIV(N, BM));

    const dim3 block(THREADS);

    if (N % BM == 0 &&
        N % BN == 0 &&
        N % BK == 0)
    {
        sgemm_block2d_prefetch<true>
            <<<grid, block>>>(
                N, N, N,
                1.f, A, B, 0.f, C);
    }
    else
    {

        sgemm_block2d_prefetch<false>
            <<<grid, block>>>(
                N, N, N,
                1.f, A, B, 0.f, C);
    }

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// CPU REFERENCE VERIFICATION
// ============================================================

static void verify_result(
    const std::vector<float> &A,
    const std::vector<float> &B,
    const std::vector<float> &C,
    int N)
{
    float maxError = 0.0f;

    for (int i = 0; i < N; ++i)
    {

        for (int j = 0; j < N; ++j)
        {

            double sum = 0.0;

            for (int k = 0; k < N; ++k)
            {

                sum +=
                    static_cast<double>(
                        A[static_cast<size_t>(i) * N + k]) *
                    static_cast<double>(
                        B[static_cast<size_t>(k) * N + j]);
            }

            const size_t idx =
                static_cast<size_t>(i) * N + j;

            const float expected =
                static_cast<float>(sum);

            const float error =
                std::fabs(C[idx] - expected);

            const float tolerance =
                1e-3f * std::max(
                            1.0f, std::fabs(expected));

            maxError = std::max(maxError, error);

            if (!std::isfinite(C[idx]) ||
                error > tolerance)
            {
                std::printf(
                    "Verification: FAIL\n");

                std::printf(
                    "C[%d][%d]: GPU=%f, CPU=%f\n",
                    i, j, C[idx], expected);

                std::exit(EXIT_FAILURE);
            }
        }
    }

    std::printf(
        "Verification: PASS (max abs error: %.6f)\n",
        maxError);
}

// ============================================================
// BENCHMARK HARNESS
// ============================================================

int main(int argc, char **argv)
{
    const int N =
        argc > 1 ? std::atoi(argv[1]) : 512;

    const int runs =
        argc > 2 ? std::atoi(argv[2]) : 5;

    const int verify =
        argc > 3 ? std::atoi(argv[3]) : 1;

    if (N <= 0 || runs <= 0 ||
        (verify != 0 && verify != 1))
    {
        std::printf(
            "Usage: block2d_prefetch.exe N runs verify(0/1)\n");

        return EXIT_FAILURE;
    }

    const size_t elements =
        static_cast<size_t>(N) * N;

    const size_t bytes =
        elements * sizeof(float);

    std::vector<float> hA(elements);
    std::vector<float> hB(elements);
    std::vector<float> hC(elements, 0.0f);

    std::mt19937 rng(42);

    std::uniform_real_distribution<float>
        dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        hA[i] = dist(rng);
        hB[i] = dist(rng);
    }

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&dA), bytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&dB), bytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&dC), bytes));

    CUDA_CHECK(cudaMemcpy(
        dA, hA.data(), bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        dB, hB.data(), bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    const dim3 grid(
        CEIL_DIV(N, BN),
        CEIL_DIV(N, BM));

    std::printf(
        "Kernel: sgemm_block2d_prefetch\n");

    std::printf("Matrix: %d x %d\n", N, N);
    std::printf("Grid: (%u, %u)\n", grid.x, grid.y);
    std::printf("Block: %d threads\n", THREADS);

    std::printf(
        "Tile: %dx%dx%d | TM=%d TN=%d\n",
        BM, BN, BK, TM, TN);

    std::printf(
        "SMEM: %zu bytes\n",
        static_cast<size_t>(
            BM * (BK + 1) + BK * BN) *
            sizeof(float));

    // Warmup.
    launch_sgemm(N, dA, dB, dC);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times;
    times.reserve(runs);

    const double operations =
        2.0 * N * N * N / 1e9;

    for (int r = 0; r < runs; ++r)
    {

        CUDA_CHECK(cudaEventRecord(start));

        launch_sgemm(N, dA, dB, dC);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.f;

        CUDA_CHECK(cudaEventElapsedTime(
            &ms, start, stop));

        times.push_back(ms);

        std::printf(
            "Run %d: %.4f ms | %.4f GFLOP/s\n",
            r + 1, ms,
            operations / (ms / 1000.0));
    }

    std::sort(times.begin(), times.end());

    const float medianMs = times[runs / 2];

    std::printf("\n===== Final Result =====\n");

    std::printf("Matrix     : %d x %d\n", N, N);
    std::printf("Operations : %.4f GFLOPs\n", operations);
    std::printf("Median time: %.4f ms\n", medianMs);

    std::printf(
        "Median perf: %.4f GFLOP/s\n",
        operations / (medianMs / 1000.0));

    if (verify)
    {

        CUDA_CHECK(cudaMemcpy(
            hC.data(), dC, bytes,
            cudaMemcpyDeviceToHost));

        verify_result(hA, hB, hC, N);
    }
    else
    {

        std::printf("Verification: SKIPPED\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
