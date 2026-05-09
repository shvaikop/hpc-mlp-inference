#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <format>
#include "FlatMatrix.hpp"

namespace MatrixGenerator {
    /**
     * Generates a FlatMatrix with controlled sparsity and value range.
     * sparsity = 0.95 means 95% of the values are zero.
     */
    template <typename T>
    FlatMatrix<T> generate(size_t rows, size_t cols, float sparsity, T min_val, T max_val) {
        std::vector<T> data;
        data.reserve(rows * cols);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> val_dist(min_val, max_val);
        std::bernoulli_distribution zero_dist(sparsity); // true = make it zero

        for (size_t i = 0; i < rows * cols; ++i) {
            if (zero_dist(gen)) {
                data.push_back(T{0});
            } else {
                data.push_back(val_dist(gen));
            }
        }
        return FlatMatrix<T>(rows, cols, data);
    }
}

// Helper to run the actual timer
void run_timer(const std::string& label, const FlatMatrix<float>& A, const FlatMatrix<float>& B, int iterations = 10) {
    FlatMatrix<float> C(A.rows(), B.rows()); // Output is M x N

    // Warm-up
    A.multiply_transposed_rhs(B, C);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        A.multiply_transposed_rhs(B, C);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = (end - start) / iterations;
    std::cout << std::format("[{:^20}] Average Time: {:.3f} ms\n", label, elapsed.count());
}

/* --- The Four Specific Benchmark Functions --- */

void benchmark_dense_by_dense(size_t M, size_t K, size_t N) {
    auto A = MatrixGenerator::generate<float>(M, K, 0.0f, -1.0f, 1.0f);
    auto B = MatrixGenerator::generate<float>(N, K, 0.0f, -1.0f, 1.0f);
    run_timer("Dense x Dense", A, B);
}

void benchmark_dense_by_sparse(size_t M, size_t K, size_t N) {
    auto A = MatrixGenerator::generate<float>(M, K, 0.0f, -1.0f, 1.0f);
    auto B = MatrixGenerator::generate<float>(N, K, 0.95f, -1.0f, 1.0f); // 95% zeros
    run_timer("Dense x Sparse", A, B);
}

void benchmark_sparse_by_dense(size_t M, size_t K, size_t N) {
    auto A = MatrixGenerator::generate<float>(M, K, 0.95f, -1.0f, 1.0f); // 95% zeros
    auto B = MatrixGenerator::generate<float>(N, K, 0.0f, -1.0f, 1.0f);
    run_timer("Sparse x Dense", A, B);
}

void benchmark_sparse_by_sparse(size_t M, size_t K, size_t N) {
    auto A = MatrixGenerator::generate<float>(M, K, 0.95f, -1.0f, 1.0f); // 95% zeros
    auto B = MatrixGenerator::generate<float>(N, K, 0.95f, -1.0f, 1.0f); // 95% zeros
    run_timer("Sparse x Sparse", A, B);
}

int main() {
    // These dimensions simulate a typical small hidden layer
    const size_t M = 16;   // Batch size
    const size_t K = 4096; // Input features
    const size_t N = 4096; // Output features (transposed rows)

    std::cout << "Starting FlatMatrix Benchmarks...\n";
    std::cout << "Note: All should have similar runtimes because FlatMatrix is a Dense storage format.\n";
    std::cout << "----------------------------------------------------------\n";

    benchmark_dense_by_dense(M, K, N);
    benchmark_dense_by_sparse(M, K, N);
    benchmark_sparse_by_dense(M, K, N);
    benchmark_sparse_by_sparse(M, K, N);

    return 0;
}
