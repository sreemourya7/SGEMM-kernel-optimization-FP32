
// Kernel 04: 1D Block Tiling (FP32 SGEMM)
// BM=64, BN=64, BK=8, TM=8
// GPU: NVIDIA GTX 1650 (sm_75)

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;

constexpr int THREADS = BM * BN / TM; // 512 threads/block

static_assert(BM % TM == 0, "BM must be divisible by TM");
static_assert(THREADS == BM * BK, "A tile loading mismatch");
static_assert(THREADS == BN * BK, "B tile loading mismatch");

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHECK(call)                                   \
    do                                                     \
    {                                                      \
        cudaError_t err = (call);                          \
        if (err != cudaSuccess)                            \
        {                                                  \
            fprintf(stderr, "CUDA error at line %d: %s\n", \
                    __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                            \
        }                                                  \
    } while (0)

// ============================================================
// Kernel 04: 1D Block Tiling
// ============================================================

__global__ void sgemm_block1d(
    int M, int N, int K,
    float alpha,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float beta,
    float *__restrict__ C)
{
    // 1024 floats = 4096 bytes = 4 KB
    __shared__ float As[BM * BK]; // 64 x 8
    __shared__ float Bs[BK * BN]; // 8 x 64

    const int tid = threadIdx.x;

    // Each thread handles 8 rows of one output column.
    const int threadCol = tid % BN;
    const int threadRow = tid / BN;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int outRow0 = blockRow + threadRow * TM;
    const int outCol = blockCol + threadCol;

    // Eight independent accumulators stored in registers.
    float acc[TM] = {0.0f};

    // Each of the 512 threads loads one element of A
    // and one element of B into shared memory.
    const int aLocalRow = tid / BK;
    const int aLocalK = tid % BK;

    const int bLocalK = tid / BN;
    const int bLocalCol = tid % BN;

    // ========================================================
    // Loop over K tiles
    // ========================================================

    for (int bk = 0; bk < K; bk += BK)
    {

        // Step 1: Load A tile from global to shared memory.
        const int globalARow = blockRow + aLocalRow;
        const int globalAK = bk + aLocalK;

        As[tid] = (globalARow < M && globalAK < K)
                      ? A[static_cast<size_t>(globalARow) * K + globalAK]
                      : 0.0f;

        // Step 2: Load B tile from global to shared memory.
        const int globalBK = bk + bLocalK;
        const int globalBCol = blockCol + bLocalCol;

        Bs[tid] = (globalBK < K && globalBCol < N)
                      ? B[static_cast<size_t>(globalBK) * N + globalBCol]
                      : 0.0f;

        __syncthreads();

        // ====================================================
        // Step 3: Compute 8 output elements per thread
        // ====================================================

#pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {

            // Load B once and reuse across 8 FMAs.
            const float b = Bs[dotIdx * BN + threadCol];

#pragma unroll
            for (int resIdx = 0; resIdx < TM; ++resIdx)
            {

                const float a =
                    As[(threadRow * TM + resIdx) * BK + dotIdx];

                acc[resIdx] = fmaf(a, b, acc[resIdx]);
            }
        }

        // Ensure every thread finishes reading shared memory
        // before the next tile overwrites it.
        __syncthreads();
    }

    // ========================================================
    // Step 4: Write 8 results to global memory
    // ========================================================

#pragma unroll
    for (int resIdx = 0; resIdx < TM; ++resIdx)
    {

        const int row = outRow0 + resIdx;

        if (row < M && outCol < N)
        {

            const size_t idx =
                static_cast<size_t>(row) * N + outCol;

            C[idx] = alpha * acc[resIdx] + beta * C[idx];
        }
    }
}

// ============================================================
// CPU Verification
// ============================================================

void verify_result(
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

            const float expected = static_cast<float>(sum);
            const float error = std::fabs(C[idx] - expected);

            const float tolerance =
                1e-3f * std::max(1.0f, std::fabs(expected));

            maxError = std::max(maxError, error);

            if (!std::isfinite(C[idx]) || error > tolerance)
            {
                printf("Verification: FAIL\n");
                printf("C[%d][%d]: GPU=%f, CPU=%f\n",
                       i, j, C[idx], expected);

                exit(EXIT_FAILURE);
            }
        }
    }

    printf("Verification: PASS (max abs error: %.6f)\n",
           maxError);
}

// ============================================================
// Main: Benchmark and Verification
// ============================================================

int main(int argc, char **argv)
{
    const int N = (argc > 1) ? atoi(argv[1]) : 512;
    const int runs = (argc > 2) ? atoi(argv[2]) : 5;
    const int verify = (argc > 3) ? atoi(argv[3]) : 1;

    if (N <= 0 || runs <= 0 || (verify != 0 && verify != 1))
    {
        printf("Usage: block1d_gemm.exe N runs verify(0/1)\n");
        return EXIT_FAILURE;
    }

    const size_t elements = static_cast<size_t>(N) * N;
    const size_t bytes = elements * sizeof(float);

    std::vector<float> hA(elements);
    std::vector<float> hB(elements);
    std::vector<float> hC(elements, 0.0f);

    // Reproducible random inputs.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

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
        dA, hA.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const dim3 block(THREADS);

    const dim3 grid(
        CEIL_DIV(N, BN),
        CEIL_DIV(N, BM));

    printf("Kernel: sgemm_block1d\n");
    printf("Matrix: %d x %d\n", N, N);
    printf("Grid: (%u, %u)\n", grid.x, grid.y);
    printf("Block: %d threads\n", THREADS);
    printf("Tile: %dx%dx%d | TM=%d\n", BM, BN, BK, TM);
    printf("SMEM: %zu bytes\n",
           static_cast<size_t>(BM * BK + BK * BN) * sizeof(float));

    // Warmup.
    sgemm_block1d<<<grid, block>>>(
        N, N, N, alpha, dA, dB, beta, dC);

    CUDA_CHECK(cudaGetLastError());
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

        sgemm_block1d<<<grid, block>>>(
            N, N, N, alpha, dA, dB, beta, dC);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &ms, start, stop));

        times.push_back(ms);

        printf("Run %d: %.4f ms | %.4f GFLOP/s\n",
               r + 1,
               ms,
               operations / (ms / 1000.0));
    }

    std::sort(times.begin(), times.end());

    const float medianMs = times[runs / 2];

    const double gflops =
        operations / (medianMs / 1000.0);

    printf("\n===== Final Result =====\n");
    printf("Matrix     : %d x %d\n", N, N);
    printf("Operations : %.4f GFLOPs\n", operations);
    printf("Median time: %.4f ms\n", medianMs);
    printf("Median perf: %.4f GFLOP/s\n", gflops);

    if (verify)
    {

        CUDA_CHECK(cudaMemcpy(
            hC.data(), dC, bytes,
            cudaMemcpyDeviceToHost));

        verify_result(hA, hB, hC, N);
    }
    else
    {
        printf("Verification: SKIPPED\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
