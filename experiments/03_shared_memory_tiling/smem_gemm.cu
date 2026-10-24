// This implementation also handles matrix sizes that aren't multiples of 32 by zero-padding incomplete tiles.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define BLOCKSIZE 32
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

// Kernel 03: Shared Memory Tiling
// Each 32x32 block computes one 32x32 tile of C.
__global__ void sgemm_smem(
    int M, int N, int K,
    float alpha,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float beta,
    float *__restrict__ C)
{
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    const int tx = threadIdx.x; // Column within tile
    const int ty = threadIdx.y; // Row within tile

    const int row = blockIdx.x * BLOCKSIZE + ty;
    const int col = blockIdx.y * BLOCKSIZE + tx;

    float tmp = 0.0f;

    // Move A horizontally and B vertically along K.
    for (int bk = 0; bk < K; bk += BLOCKSIZE)
    {

        // Cooperatively load A and B into shared memory.
        // Adjacent threads access consecutive global addresses.
        As[ty][tx] = (row < M && bk + tx < K)
                         ? A[(size_t)row * K + bk + tx]
                         : 0.0f;

        Bs[ty][tx] = (bk + ty < K && col < N)
                         ? B[(size_t)(bk + ty) * N + col]
                         : 0.0f;

        // Wait until both tiles are completely loaded.
        __syncthreads();

// Reuse shared-memory tiles for 32 dot-product steps.
#pragma unroll
        for (int k = 0; k < BLOCKSIZE; ++k)
        {
            tmp += As[ty][k] * Bs[k][tx];
        }

        // Ensure all threads finish using the current tiles.
        __syncthreads();
    }

    // Write the final output.
    if (row < M && col < N)
    {
        const size_t idx = (size_t)row * N + col;
        C[idx] = alpha * tmp + beta * C[idx];
    }
}

// CPU correctness verification
void verify_result(
    const std::vector<float> &A,
    const std::vector<float> &B,
    const std::vector<float> &C,
    int N)
{
    float max_error = 0.0f;

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            double sum = 0.0;

            for (int k = 0; k < N; ++k)
            {
                sum += (double)A[(size_t)i * N + k] * (double)B[(size_t)k * N + j];
            }

            const size_t idx = (size_t)i * N + j;
            const float expected = (float)sum;
            const float error = std::fabs(C[idx] - expected);
            const float tolerance =
                1e-3f * std::max(1.0f, std::fabs(expected));

            max_error = std::max(max_error, error);

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
           max_error);
}

int main(int argc, char **argv)
{
    const int N = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int runs = (argc > 2) ? std::atoi(argv[2]) : 5;
    const int verify = (argc > 3) ? std::atoi(argv[3]) : 1;

    if (N <= 0 || runs <= 0 || (verify != 0 && verify != 1))
    {
        printf("Usage: smem_gemm.exe N runs verify(0/1)\n");
        return EXIT_FAILURE;
    }

    const size_t elements = (size_t)N * N;
    const size_t bytes = elements * sizeof(float);

    std::vector<float> h_A(elements);
    std::vector<float> h_B(elements);
    std::vector<float> h_C(elements, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        h_A[i] = dist(rng);
        h_B[i] = dist(rng);
    }

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(
        CEIL_DIV(N, BLOCKSIZE),
        CEIL_DIV(N, BLOCKSIZE));

    printf("Kernel: sgemm_smem\n");
    printf("Matrix: %d x %d\n", N, N);
    printf("Grid: (%u, %u)\n", gridDim.x, gridDim.y);
    printf("Block: %u threads\n",
           blockDim.x * blockDim.y);
    printf("Tile: %d x %d\n", BLOCKSIZE, BLOCKSIZE);

    // Warmup
    sgemm_smem<<<gridDim, blockDim>>>(
        N, N, N, alpha, d_A, d_B, beta, d_C);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times;
    times.reserve(runs);

    const double operations = 2.0 * N * N * N / 1e9;

    for (int r = 0; r < runs; ++r)
    {
        CUDA_CHECK(cudaEventRecord(start));

        sgemm_smem<<<gridDim, blockDim>>>(
            N, N, N, alpha, d_A, d_B, beta, d_C);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        times.push_back(ms);

        printf("Run %d: %.4f ms | %.4f GFLOP/s\n",
               r + 1, ms, operations / (ms / 1000.0));
    }

    std::sort(times.begin(), times.end());

    const float median_ms = times[runs / 2];
    const double gflops = operations / (median_ms / 1000.0);

    printf("\n===== Final Result =====\n");
    printf("Matrix     : %d x %d\n", N, N);
    printf("Operations : %.4f GFLOPs\n", operations);
    printf("Median time: %.4f ms\n", median_ms);
    printf("Median perf: %.4f GFLOP/s\n", gflops);

    if (verify)
    {
        CUDA_CHECK(cudaMemcpy(
            h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

        verify_result(h_A, h_B, h_C, N);
    }
    else
    {
        printf("Verification: SKIPPED\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
