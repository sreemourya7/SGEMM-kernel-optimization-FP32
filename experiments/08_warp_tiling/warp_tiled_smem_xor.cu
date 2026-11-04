
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Kernel 08: Warp-tiled FP32 SGEMM for GTX 1650 / sm_75.
// Each warp computes an explicit WM x WN tile,
// split into WMITER row subtiles.
//
// Keeps Kernel 05's global-to-shared loading,
// padded A layout, and vectorized B/C operations.
//
// Tune warp layout with -DLANE_COLS=4, 8, or 16.
//
// Row-major C = alpha * (A @ B) + beta * C.

// ============================================================
// 1. BLOCK TILE
// ============================================================

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 16;

constexpr int THREADS = 256;

// ============================================================
// 2. WARP AND THREAD TILING
// ============================================================

// Each thread computes two 4x4 subtiles:
// 2 * 4 * 4 = 32 output elements.

constexpr int TM = 4;
constexpr int TN = 4;

constexpr int WMITER = 2;
constexpr int WNITER = 1;

// Number of lane groups along N.
// Default: 8 -> 4 x 8 lane mapping.

#ifndef LANE_COLS
#define LANE_COLS 8
#endif

constexpr int LANE_N = LANE_COLS;
constexpr int LANE_M = 32 / LANE_N;

// Warp subtile dimensions.
constexpr int WSUBM = LANE_M * TM;
constexpr int WSUBN = LANE_N * TN;

// Full warp tile dimensions.
constexpr int WM = WMITER * WSUBM;
constexpr int WN = WNITER * WSUBN;

// Number of warps along output dimensions.
constexpr int WARP_COLS = BN / WN;
constexpr int WARP_ROWS = BM / WM;

constexpr int NUM_WARPS = THREADS / 32;

// Shared-memory loading constants.
constexpr int B_COL_VECTORS = BN / 4;
constexpr int B_VECTORS = BK * B_COL_VECTORS;
constexpr int A_LOADS = (BM * BK) / THREADS;

static_assert(
    LANE_N == 4 || LANE_N == 8 || LANE_N == 16,
    "Use LANE_COLS=4, 8, or 16");

static_assert(
    BM % WM == 0 && BN % WN == 0,
    "Invalid warp tile");

static_assert(
    WARP_ROWS * WARP_COLS == NUM_WARPS,
    "Wrong warp coverage");

static_assert(
    (BM * BK) % THREADS == 0,
    "Uneven A loading");

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
// 3. WARP-TILED SGEMM KERNEL
// ============================================================

template <bool FULL>
__global__ void sgemm_warp_tiled(
    int M, int N, int K,
    float alpha,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float beta,
    float *__restrict__ C)
{
    // Preserve Kernel 05's shared-memory layout.
    __shared__ float As[BM][BK];

    __shared__ float4 Bs[BK][B_COL_VECTORS];

    // --------------------------------------------------------
    // THREAD -> WARP -> OUTPUT MAPPING
    // --------------------------------------------------------

    const int tid = threadIdx.x;

    const int warpId = tid / 32;
    const int laneId = tid % 32;

    const int warpRow = warpId / WARP_COLS;
    const int warpCol = warpId % WARP_COLS;

    const int laneRow = laneId / LANE_N;
    const int laneCol = laneId % LANE_N;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int warpBaseRow = blockRow + warpRow * WM;
    const int warpBaseCol = blockCol + warpCol * WN;

    // Register-resident output accumulators.
    // 2 * 1 * 4 * 4 = 32 FP32 accumulators.
    float acc[WMITER][WNITER][TM][TN] = {};

    // ========================================================
    // 4. LOOP OVER K-TILES
    // ========================================================

    for (int bk = 0; bk < K; bk += BK)
    {

        // ----------------------------------------------------
        // A: GLOBAL MEMORY -> SHARED MEMORY
        // Original coalesced scalar loading.
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

                As[aRow][aK ^ ((aRow / TM) & (BK - 1))] =
                    A[static_cast<size_t>(globalRow) * K + globalK];
            }
            else
            {

                As[aRow][aK ^ ((aRow / TM) & (BK - 1))] =
                    (globalRow < M && globalK < K)
                        ? A[static_cast<size_t>(globalRow) * K + globalK]
                        : 0.0f;
            }
        }

        // ----------------------------------------------------
        // B: GLOBAL MEMORY -> SHARED MEMORY
        // Vectorized float4 loading.
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
        // 5. WARP-TILE OUTER PRODUCT
        // ====================================================

#pragma unroll 8
        for (int k = 0; k < BK; ++k)
        {

            // Register fragments for this lane's
            // assigned warp subtiles.

            float regA[WMITER][TM];
            float4 regB[WNITER];

            // -----------------------------------------------
            // A: SHARED -> REGISTERS
            // -----------------------------------------------

#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi)
            {

#pragma unroll
                for (int i = 0; i < TM; ++i)
                {

                    const int aRow =
                        warpRow * WM + mi * WSUBM + laneRow * TM + i;

                    regA[mi][i] =
    As[aRow][k ^ ((aRow / TM) & (BK - 1))];
                }
            }

            // -----------------------------------------------
            // B: SHARED -> REGISTERS
            // -----------------------------------------------

#pragma unroll
            for (int ni = 0; ni < WNITER; ++ni)
            {

                const int col =
                    warpCol * WN + ni * WSUBN + laneCol * TN;

                regB[ni] = Bs[k][col / 4];
            }

            // -----------------------------------------------
            // COMPUTE WARP SUBTILES
            // -----------------------------------------------

#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi)
            {

#pragma unroll
                for (int ni = 0; ni < WNITER; ++ni)
                {

                    const float4 b = regB[ni];

#pragma unroll
                    for (int i = 0; i < TM; ++i)
                    {

                        const float a = regA[mi][i];

                        acc[mi][ni][i][0] =
                            fmaf(a, b.x,
                                 acc[mi][ni][i][0]);

                        acc[mi][ni][i][1] =
                            fmaf(a, b.y,
                                 acc[mi][ni][i][1]);

                        acc[mi][ni][i][2] =
                            fmaf(a, b.z,
                                 acc[mi][ni][i][2]);

                        acc[mi][ni][i][3] =
                            fmaf(a, b.w,
                                 acc[mi][ni][i][3]);
                    }
                }
            }
        }

        // Protect shared-memory tiles before overwrite.
        __syncthreads();
    }

    // ========================================================
    // 6. REGISTER ACCUMULATORS -> GLOBAL C
    // ========================================================

#pragma unroll
    for (int mi = 0; mi < WMITER; ++mi)
    {

#pragma unroll
        for (int ni = 0; ni < WNITER; ++ni)
        {

            const int col =
                warpBaseCol + ni * WSUBN + laneCol * TN;

#pragma unroll
            for (int i = 0; i < TM; ++i)
            {

                const int row =
                    warpBaseRow + mi * WSUBM + laneRow * TM + i;

                if constexpr (FULL)
                {

                    const size_t pos =
                        static_cast<size_t>(row) * N + col;

                    float4 out = make_float4(
                        alpha * acc[mi][ni][i][0],
                        alpha * acc[mi][ni][i][1],
                        alpha * acc[mi][ni][i][2],
                        alpha * acc[mi][ni][i][3]);

                    float4 *dst =
                        reinterpret_cast<float4 *>(
                            C + pos);

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

                        if (col + j < N)
                        {

                            const size_t pos =
                                static_cast<size_t>(row) * N + col + j;

                            const float value =
                                alpha * acc[mi][ni][i][j];

                            C[pos] = (beta == 0.0f)
                                         ? value
                                         : value + beta * C[pos];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================
// 7. KERNEL LAUNCHER
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
        sgemm_warp_tiled<true>
            <<<grid, block>>>(
                N, N, N, 1.f, A, B, 0.f, C);
    }
    else
    {

        sgemm_warp_tiled<false>
            <<<grid, block>>>(
                N, N, N, 1.f, A, B, 0.f, C);
    }

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// 8. CPU REFERENCE VERIFICATION
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

// ============================================================
// 9. BENCHMARK HARNESS
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
            "Usage: warp_tiled_gemm.exe N runs verify(0/1)\n");
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
        "Kernel: sgemm_warp_tiled\n");

    std::printf("Matrix: %d x %d\n", N, N);
    std::printf("Grid: (%u, %u)\n", grid.x, grid.y);
    std::printf("Block: %d threads\n", THREADS);

    std::printf(
        "Tile: %dx%dx%d | Warp tile: %dx%d\n",
        BM, BN, BK, WM, WN);

    std::printf(
        "Lane layout: %dx%d | Outputs/thread: %d\n",
        LANE_M, LANE_N, TM * TN * WMITER * WNITER);

    std::printf(
        "SMEM: %zu bytes\n",
        static_cast<size_t>(
            BM * BK + BK * BN) *
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
