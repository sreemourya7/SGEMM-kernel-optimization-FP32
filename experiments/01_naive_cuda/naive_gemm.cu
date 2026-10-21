#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)


// ============================================================
// Kernel 01: Naive CUDA SGEMM
//
// One CUDA thread computes one element of C.
//
// C = alpha * (A @ B) + beta * C
// ============================================================

__global__ void sgemm_naive(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C)
{
    // x -> row of C
    // y -> column of C
    const unsigned int x =
        blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int y =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (x < static_cast<unsigned int>(M) &&
        y < static_cast<unsigned int>(N))
    {
        float tmp = 0.0f;

        for (int i = 0; i < K; ++i) {
            tmp += A[static_cast<size_t>(x) * K + i] *
                   B[static_cast<size_t>(i) * N + y];
        }

        C[static_cast<size_t>(x) * N + y] =
            alpha * tmp +
            beta * C[static_cast<size_t>(x) * N + y];
    }
}


// ============================================================
// FP64-accumulated CPU reference
// Used only for correctness verification.
// ============================================================

void gemm_reference(
    const float* A,
    const float* B,
    double* C,
    size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {

            double sum = 0.0;

            for (size_t k = 0; k < N; ++k) {
                sum += static_cast<double>(A[i * N + k]) *
                       static_cast<double>(B[k * N + j]);
            }

            C[i * N + j] = sum;
        }
    }
}


// ============================================================
// Full-matrix correctness check
// ============================================================

bool verify_result(
    const double* reference,
    const float* result,
    size_t elements)
{
    constexpr double REL_TOL = 1e-3;

    for (size_t i = 0; i < elements; ++i) {

        const double ref =
            reference[i];

        const double value =
            static_cast<double>(result[i]);

        const double diff =
            std::abs(ref - value);

        const double tolerance =
            REL_TOL * std::max(1.0, std::abs(ref));

        if (diff > tolerance) {

            std::cerr
                << "\nVerification FAILED\n"
                << "Index      : " << i << '\n'
                << "Reference  : " << ref << '\n'
                << "GPU result : " << value << '\n'
                << "Difference : " << diff << '\n'
                << "Tolerance  : " << tolerance << '\n';

            return false;
        }
    }

    return true;
}


// ============================================================
// Median helper
// ============================================================

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());

    const size_t n = values.size();

    if (n % 2 == 1)
        return values[n / 2];

    return 0.5 *
           (values[n / 2 - 1] + values[n / 2]);
}


// ============================================================
// Main benchmark
// ============================================================

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cout
            << "Usage:\n"
            << "  01_naive_cuda.exe <N> <runs> <verify>\n\n"
            << "Examples:\n"
            << "  01_naive_cuda.exe 512 5 1\n"
            << "  01_naive_cuda.exe 4096 5 0\n\n"
            << "verify = 1 -> full FP64 CPU verification\n"
            << "verify = 0 -> benchmark only\n";

        return 1;
    }

    size_t N;
    int numRuns;
    int verify;

    try {
        N = std::stoull(argv[1]);
        numRuns = std::stoi(argv[2]);
        verify = std::stoi(argv[3]);
    }
    catch (...) {
        std::cerr << "Invalid command-line arguments.\n";
        return 1;
    }

    if (N == 0 ||
        N > static_cast<size_t>(INT_MAX) ||
        numRuns <= 0 ||
        (verify != 0 && verify != 1))
    {
        std::cerr << "Invalid input.\n";
        return 1;
    }

    const int n = static_cast<int>(N);

    if (N > SIZE_MAX / N) {
        std::cerr << "Matrix size overflow.\n";
        return 1;
    }

    const size_t elements =
        N * N;

    const size_t bytes =
        elements * sizeof(float);

    // --------------------------------------------------------
    // GPU information
    // --------------------------------------------------------

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "\n===== Device =====\n";
    std::cout << "GPU          : " << prop.name << '\n';

    // --------------------------------------------------------
    // Host matrices
    // --------------------------------------------------------

    std::vector<float> h_A(elements);
    std::vector<float> h_B(elements);
    std::vector<float> h_C(elements, 0.0f);

    // Fixed seed -> identical input across kernel versions.
    std::mt19937 generator(42);

    std::uniform_real_distribution<float>
        distribution(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i) {
        h_A[i] = distribution(generator);
        h_B[i] = distribution(generator);
    }

    // --------------------------------------------------------
    // Optional CPU reference
    // --------------------------------------------------------

    std::vector<double> reference;

    if (verify) {
        reference.resize(elements);

        std::cout
            << "\nComputing FP64 CPU reference...\n";

        gemm_reference(
            h_A.data(),
            h_B.data(),
            reference.data(),
            N);
    }

    // --------------------------------------------------------
    // Device allocation
    // --------------------------------------------------------

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A.data(),
        bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B.data(),
        bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(
        d_C,
        0,
        bytes));

    // --------------------------------------------------------
    // Kernel configuration
    // --------------------------------------------------------

    dim3 blockDim(32, 32, 1);

    dim3 gridDim(
        CEIL_DIV(n, 32),
        CEIL_DIV(n, 32),
        1);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    std::cout << "\n===== Benchmark Configuration =====\n";

    std::cout
        << "Matrix       : "
        << N << " x " << N << '\n';

    std::cout
        << "Block        : "
        << blockDim.x << " x "
        << blockDim.y << '\n';

    std::cout
        << "Grid         : "
        << gridDim.x << " x "
        << gridDim.y << '\n';

    std::cout
        << "Timed runs   : "
        << numRuns << '\n';

    std::cout
        << "Random seed  : 42\n";

    std::cout
        << "Verification : "
        << (verify ? "enabled" : "disabled")
        << '\n';

    // --------------------------------------------------------
    // Warm-up
    // --------------------------------------------------------

    std::cout << "\nRunning GPU warm-up...\n";

    sgemm_naive<<<gridDim, blockDim>>>(
        n,
        n,
        n,
        alpha,
        d_A,
        d_B,
        beta,
        d_C);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify warm-up.
    if (verify) {

        CUDA_CHECK(cudaMemcpy(
            h_C.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost));

        if (!verify_result(
                reference.data(),
                h_C.data(),
                elements))
        {
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));

            return 1;
        }

        std::cout
            << "Warm-up verification: PASS\n";
    }

    // --------------------------------------------------------
    // FLOP count
    // --------------------------------------------------------

    const double totalFlops =
        2.0 *
        static_cast<double>(N) *
        static_cast<double>(N) *
        static_cast<double>(N);

    // --------------------------------------------------------
    // CUDA timing events
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> timesMs;
    timesMs.reserve(numRuns);

    std::cout << "\n===== Timed Runs =====\n";

    for (int run = 0; run < numRuns; ++run) {

        CUDA_CHECK(cudaEventRecord(start));

        sgemm_naive<<<gridDim, blockDim>>>(
            n,
            n,
            n,
            alpha,
            d_A,
            d_B,
            beta,
            d_C);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds,
            start,
            stop));

        const double seconds =
            static_cast<double>(milliseconds) / 1000.0;

        const double gflops =
            totalFlops / seconds / 1e9;

        timesMs.push_back(milliseconds);

        std::cout
            << std::fixed
            << std::setprecision(4)
            << "Run " << run + 1
            << ": "
            << milliseconds << " ms"
            << " | "
            << gflops << " GFLOP/s\n";
    }

    // --------------------------------------------------------
    // Final correctness check
    // --------------------------------------------------------

    if (verify) {

        CUDA_CHECK(cudaMemcpy(
            h_C.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost));

        if (!verify_result(
                reference.data(),
                h_C.data(),
                elements))
        {
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            return 1;
        }
    }

    // --------------------------------------------------------
    // Median result
    // --------------------------------------------------------

    const double medianMs =
        median(timesMs);

    const double medianSeconds =
        medianMs / 1000.0;

    const double medianGFLOPS =
        totalFlops /
        medianSeconds /
        1e9;

    std::cout
        << "\n===== Final Result =====\n";

    std::cout
        << "Matrix       : "
        << N << " x " << N << '\n';

    std::cout
        << "Operations   : "
        << totalFlops / 1e9
        << " GFLOPs\n";

    std::cout
        << "Median time  : "
        << medianMs
        << " ms\n";

    std::cout
        << "Median perf. : "
        << medianGFLOPS
        << " GFLOP/s\n";

    if (verify) {
        std::cout
            << "Correctness  : PASS\n";
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
