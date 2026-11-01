
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                       \
    do                                                         \
    {                                                          \
        cudaError_t e = (call);                                \
        if (e != cudaSuccess)                                  \
        {                                                      \
            printf("CUDA error: %s\n", cudaGetErrorString(e)); \
            std::exit(1);                                      \
        }                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                        \
    do                                            \
    {                                             \
        cublasStatus_t s = (call);                \
        if (s != CUBLAS_STATUS_SUCCESS)           \
        {                                         \
            printf("cuBLAS error: %d\n", (int)s); \
            std::exit(1);                         \
        }                                         \
    } while (0)

int main(int argc, char **argv)
{
    int N = argc > 1 ? std::atoi(argv[1]) : 4096;
    int runs = argc > 2 ? std::atoi(argv[2]) : 9;

    if (N <= 0 || runs <= 0)
        return 1;

    size_t count = (size_t)N * N;
    size_t bytes = count * sizeof(float);

    std::vector<float> A(count), B(count);

    for (size_t i = 0; i < count; ++i)
    {
        A[i] = (float)((int)(i % 17) - 8) / 16.0f;
        B[i] = (float)((int)(i % 13) - 6) / 16.0f;
    }

    float *dA, *dB, *dC;

    CUDA_CHECK(cudaMalloc((void **)&dA, bytes));
    CUDA_CHECK(cudaMalloc((void **)&dB, bytes));
    CUDA_CHECK(cudaMalloc((void **)&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes,
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major matrices.
    // Swap A and B to obtain row-major C = A * B.
    auto gemm = [&]()
    {
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            dB, N,
            dA, N,
            &beta,
            dC, N));
    };

    // Warmup
    for (int i = 0; i < 5; ++i)
        gemm();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times;

    for (int i = 0; i < runs; ++i)
    {
        CUDA_CHECK(cudaEventRecord(start));

        gemm();

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());

    // Median; average middle two for even run counts.
    float median = times[runs / 2];
    if (runs % 2 == 0)
        median = (times[runs / 2 - 1] + times[runs / 2]) / 2;

    double gflops = (2.0 * N * N * N) / (median * 1e6);

    // Basic correctness check: C[0,0] in row-major layout.
    float c00;
    CUDA_CHECK(cudaMemcpy(&c00, dC, sizeof(float),
                          cudaMemcpyDeviceToHost));

    double expected = 0.0;
    for (int k = 0; k < N; ++k)
        expected += (double)A[k] * B[(size_t)k * N];

    printf("\n===== cuBLAS SGEMM =====\n");
    printf("Matrix      : %d x %d\n", N, N);
    printf("Runs        : %d\n", runs);
    printf("Median time : %.4f ms\n", median);
    printf("Performance : %.4f GFLOP/s\n", gflops);
    printf("C[0,0]      : %.6f (reference %.6f)\n",
           c00, expected);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
