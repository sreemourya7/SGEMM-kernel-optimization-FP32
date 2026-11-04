
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Kernel 06-B: Vectorized global loads for A only.
// Keep Kernel 05's original shared-memory layout and computation.
// GTX 1650, sm_75.

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 16;

constexpr int TM = 8;
constexpr int TN = 4;

constexpr int THREADS = 256;
constexpr int COL_GROUPS = BN / TN;
constexpr int B_COL_VECTORS = BN / 4;
constexpr int B_VECTORS = BK * B_COL_VECTORS;

constexpr int A_VEC4_COUNT = BM * (BK / 4);
constexpr int A_VEC4_LOADS = A_VEC4_COUNT / THREADS;

static_assert(BK % 4 == 0, "BK must be divisible by 4");
static_assert(A_VEC4_COUNT % THREADS == 0,
              "A vector loads must divide evenly");
static_assert((BM / TM) * (BN / TN) == THREADS,
              "Invalid thread mapping");

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

template <bool FULL>
__global__ void sgemm_block2d_vecA_only(
    int M, int N, int K,
    float alpha,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float beta,
    float *__restrict__ C)
{
    // Original padded A layout from Kernel 05.
    __shared__ float As[BM][BK + 1];

    // Original vectorized B layout.
    __shared__ float4 Bs[BK][B_COL_VECTORS];

    const int tid = threadIdx.x;

    const int threadRow = tid / COL_GROUPS;
    const int threadCol = tid % COL_GROUPS;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int outRow = blockRow + threadRow * TM;
    const int outCol = blockCol + threadCol * TN;

    // 32 FP32 accumulators per thread.
    float acc[TM][TN] = {};

    for (int bk = 0; bk < K; bk += BK)
    {

        // ===================================================
        // STEP 1: A GMEM -> SMEM
        // NEW: float4 global-memory loads.
        // ===================================================

#pragma unroll
        for (int load = 0;
             load < A_VEC4_LOADS;
             ++load)
        {
            const int vec = tid + load * THREADS;

            const int aRow = vec / (BK / 4);
            const int aK = (vec % (BK / 4)) * 4;

            const int globalRow = blockRow + aRow;
            const int globalK = bk + aK;

            const size_t offset =
                static_cast<size_t>(globalRow) * K + globalK;

            float4 v;

            if constexpr (FULL)
            {

                // 16-byte aligned on the full-tile path.
                v = *reinterpret_cast<const float4 *>(
                    A + offset);
            }
            else
            {

                // Boundary-safe fallback.
                v = make_float4(0.f, 0.f, 0.f, 0.f);

                if (globalRow < M)
                {
                    if (globalK + 0 < K)
                        v.x = A[offset + 0];

                    if (globalK + 1 < K)
                        v.y = A[offset + 1];

                    if (globalK + 2 < K)
                        v.z = A[offset + 2];

                    if (globalK + 3 < K)
                        v.w = A[offset + 3];
                }
            }

            // Preserve Kernel 05's original A layout.
            // No transpose.
            As[aRow][aK + 0] = v.x;
            As[aRow][aK + 1] = v.y;
            As[aRow][aK + 2] = v.z;
            As[aRow][aK + 3] = v.w;
        }

        // ===================================================
        // STEP 2: B GMEM -> SMEM
        // Unchanged: vectorized float4 loads.
        // ===================================================

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

                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);

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

        // ===================================================
        // STEP 3: SMEM -> REGISTERS + OUTER PRODUCT
        // Unchanged from Kernel 05.
        // ===================================================

#pragma unroll
        for (int k = 0; k < BK; ++k)
        {

            const float4 b = Bs[k][threadCol];

#pragma unroll
            for (int i = 0; i < TM; ++i)
            {

                const float a =
                    As[threadRow * TM + i][k];

                acc[i][0] = fmaf(a, b.x, acc[i][0]);
                acc[i][1] = fmaf(a, b.y, acc[i][1]);
                acc[i][2] = fmaf(a, b.z, acc[i][2]);
                acc[i][3] = fmaf(a, b.w, acc[i][3]);
            }
        }

        __syncthreads();
    }

    // =======================================================
    // STEP 4: REGISTERS -> GLOBAL C
    // Unchanged: vectorized float4 stores.
    // =======================================================

#pragma unroll
    for (int i = 0; i < TM; ++i)
    {

        const int row = outRow + i;

        if constexpr (FULL)
        {

            const size_t offset =
                static_cast<size_t>(row) * N + outCol;

            float4 out = make_float4(
                alpha * acc[i][0],
                alpha * acc[i][1],
                alpha * acc[i][2],
                alpha * acc[i][3]);

            float4 *dst =
                reinterpret_cast<float4 *>(C + offset);

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

// ===========================================================
// KERNEL LAUNCH
// ===========================================================

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
        sgemm_block2d_vecA_only<true>
            <<<grid, block>>>(
                N, N, N,
                1.f, A, B, 0.f, C);
    }
    else
    {

        sgemm_block2d_vecA_only<false>
            <<<grid, block>>>(
                N, N, N,
                1.f, A, B, 0.f, C);
    }

    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================
// CPU REFERENCE VERIFICATION
// ===========================================================

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
                std::printf("Verification: FAIL\n");
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

// ===========================================================
// BENCHMARK HARNESS
// ===========================================================

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
            "Usage: block2d_vecA_only.exe N runs verify(0/1)\n");

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

    std::printf("Kernel: sgemm_block2d_vecA_only\n");
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

    const float medianMs =
        (runs % 2 == 0)
            ? 0.5f * (times[runs / 2 - 1] + times[runs / 2])
            : times[runs / 2];

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
