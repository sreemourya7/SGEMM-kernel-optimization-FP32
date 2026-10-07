/*
author :: sreemourya7
created :: 05/10/226 17:29
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <string>
//#include <bits/stdc++.h>

// FP32 GEMM used for the actual CPU benchmark.
void gemmCPU(
    const float* A,
    const float* B,
    float* C,
    size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {

            float sum = 0.0f;

            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

// Higher-precision reference implementation.
// Inputs are single-precision (FP32), but accumulation is done in double-precision (FP64).
void gemmReference(
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

bool verifyResults(
    const double* reference,
    const float* result,
    size_t elements)
{
    constexpr double REL_TOL = 1e-3;

    for (size_t i = 0; i < elements; ++i) {

        const double ref = reference[i];
        const double value = static_cast<double>(result[i]);

        const double diff = std::abs(ref - value);

        // Relative tolerance:
        // tolerance grows with the magnitude of the expected result.
        const double tolerance =
            REL_TOL * std::max(1.0, std::abs(ref));

        if (diff > tolerance) {
            std::cout << "\nVerification FAILED at index " << i
                      << "\nReference : " << ref
                      << "\nResult    : " << value
                      << "\nDifference: " << diff
                      << "\nTolerance : " << tolerance
                      << '\n';

            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cout
            << "Usage:\n"
            << "  cpu_gemm.exe <matrix_size> <runs>\n\n"
            << "Example:\n"
            << "  cpu_gemm.exe 512 3\n";

        return 1;
    }

    size_t N;
    int numRuns;

    try {
        N = std::stoull(argv[1]);
        numRuns = std::stoi(argv[2]);
    }
    catch (...) {
        std::cout << "Error: N and run count must be valid integers.\n";
        return 1;
    }

    if (N == 0 || numRuns <= 0) {
        std::cout << "Error: N and run count must be positive.\n";
        return 1;
    }

    const size_t elements = N * N;

    std::vector<float> A(elements);
    std::vector<float> B(elements);
    std::vector<float> C(elements);

    // Double-precision reference output.
    std::vector<double> reference(elements);

    // Fixed seed means every implementation sees exactly the same input.
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }

    std::cout << "Matrix size : " << N << " x " << N << '\n';
    std::cout << "Timed runs  : " << numRuns << '\n';
    std::cout << "Random seed : 42\n";

    // Generate a higher-precision reference.
    std::cout << "\nComputing FP64-accumulated reference...\n";
    gemmReference(A.data(), B.data(), reference.data(), N);

    // Untimed FP32 warm-up.
    std::cout << "Running untimed FP32 warm-up...\n";
    gemmCPU(A.data(), B.data(), C.data(), N);

    if (!verifyResults(reference.data(), C.data(), elements)) {
        return 1;
    }

    std::cout << "Warm-up verification: PASS\n";

    const double totalFlops =
        2.0 *
        static_cast<double>(N) *
        static_cast<double>(N) *
        static_cast<double>(N);

    std::vector<double> times;
    std::vector<double> performances;

    std::cout << "\n===== Timed Runs =====\n";

    for (int run = 0; run < numRuns; ++run) {

        auto start = std::chrono::steady_clock::now();

        gemmCPU(A.data(), B.data(), C.data(), N);

        auto end = std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration<double>(end - start).count();

        const double gflops =
            totalFlops / seconds / 1e9;

        if (!verifyResults(reference.data(), C.data(), elements)) {
            return 1;
        }

        times.push_back(seconds);
        performances.push_back(gflops);

        std::cout << std::fixed << std::setprecision(4)
                  << "Run " << run + 1
                  << ": " << seconds << " s"
                  << " | " << gflops << " GFLOP/s"
                  << " | PASS\n";
    }

    std::sort(times.begin(), times.end());
    std::sort(performances.begin(), performances.end());

    const double medianTime =
        times[times.size() / 2];

    const double medianGFLOPS =
        performances[performances.size() / 2];

    std::cout << "\n===== Final Result =====\n"
              << "Matrix       : " << N << " x " << N << '\n'
              << "Operations   : " << totalFlops / 1e9 << " GFLOPs\n"
              << "Median time  : " << medianTime << " seconds\n"
              << "Median perf. : " << medianGFLOPS << " GFLOP/s\n"
              << "Correctness  : PASS\n";

    return 0;
}
