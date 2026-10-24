
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

// Kernel 02: Global Memory Coalescing
__global__ void sgemm_coalescing(
    int M, int N, int K,
    float alpha,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float beta,
    float *__restrict__ C)
{
    // 32 neighboring lanes compute 32 neighboring columns.
    const int x = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;

    const int y = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

    if (x < M && y < N)
    {
        float tmp = 0.0f;

        for (int i = 0; i < K; ++i)
        {
            // A: same address across warp lanes
            // B: consecutive addresses across warp lanes
            tmp += A[(size_t)x * K + i] * B[(size_t)i * N + y];
        }

        const size_t idx = (size_t)x * N + y;
        C[idx] = alpha * tmp + beta * C[idx];
    }
}

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
    int N = (argc > 1) ? std::atoi(argv[1]) : 512;
    int runs = (argc > 2) ? std::atoi(argv[2]) : 5;
    int verify = (argc > 3) ? std::atoi(argv[3]) : 1;

    if (N <= 0 || runs <= 0 || (verify != 0 && verify != 1))
    {
        printf("Usage: gmem_gemm.exe N runs verify(0/1)\n");
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

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(),
                          bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(),
                          bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dim3 gridDim(CEIL_DIV(N, BLOCKSIZE),
                 CEIL_DIV(N, BLOCKSIZE));

    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

    printf("Kernel: sgemm_coalescing\n");
    printf("Matrix: %d x %d\n", N, N);
    printf("Grid: (%u, %u)\n", gridDim.x, gridDim.y);
    printf("Block: %u threads\n", blockDim.x);

    // Warmup
    sgemm_coalescing<<<gridDim, blockDim>>>(
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

        sgemm_coalescing<<<gridDim, blockDim>>>(
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
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C,
                              bytes, cudaMemcpyDeviceToHost));
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
