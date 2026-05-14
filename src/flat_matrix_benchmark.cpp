#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "FlatMatrix.hpp"

namespace MatrixGenerator
{
    /**
     * Generates a FlatMatrix with controlled sparsity and value range.
     * sparsity = 0.95 means 95% of the values are zero.
     */
    template <typename T>
    FlatMatrix<T> generate(size_t rows, size_t cols, float sparsity, T min_val, T max_val)
    {
        std::vector<T> data;
        data.reserve(rows * cols);

        std::mt19937 gen(42);
        std::uniform_real_distribution<T> val_dist(min_val, max_val);
        std::bernoulli_distribution zero_dist(sparsity); // true = make it zero

        for (size_t i = 0; i < rows * cols; ++i)
        {
            if (zero_dist(gen))
            {
                data.push_back(T{0});
            }
            else
            {
                data.push_back(val_dist(gen));
            }
        }
        return FlatMatrix<T>(rows, cols, data);
    }

    /**
     * Generates a vector with controlled sparsity and value range.
     * sparsity = 0.95 means 95% of the values are zero.
     */
    template <typename T>
    std::vector<float> generateVec(std::size_t rows, float sparsity, T min_val, T max_val)
    {
        std::vector<T> data;
        data.reserve(rows);

        std::mt19937 gen(42);
        std::uniform_real_distribution<T> val_dist(min_val, max_val);
        std::bernoulli_distribution zero_dist(sparsity); // true = make it zero

        for (std::size_t i = 0; i < rows; ++i)
        {
            if (zero_dist(gen))
            {
                data.push_back(T{0});
            }
            else
            {
                data.push_back(val_dist(gen));
            }
        }
        return data;
    }
}

// Helper to run the actual timer
void run_matrix_mul_benchmark(const std::string &label, const FlatMatrix<float> &A, const FlatMatrix<float> &B, int iterations = 20, std::size_t warm_up_iters = 5)
{
    FlatMatrix<float> C(A.rows(), B.rows()); // Output is M x N

    // Warm-up
    for (std::size_t i = 0; i < warm_up_iters; ++i)
    {
        A.multiply_transposed_rhs(B, C);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        A.multiply_transposed_rhs(B, C);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = (end - start) / iterations;

    float checksum = 0.0f;
    for (std::size_t r = 0; r < C.rows(); ++r)
        for (std::size_t c = 0; c < C.cols(); ++c)
            checksum += C(r, c);

    printf("[%20s] Average Time: %.3f ms  checksum: %.4f\n", label.c_str(), elapsed.count(), checksum);

    // Force the compiler to care about the result
    // This tiny check makes it impossible to optimize away the multiplication
    if (C.rows() > 0 && C(0, 0) == 999.999f)
    {
        std::cout << "This will almost never happen, but the compiler doesn't know that.\n";
    }
}

// Helper to run the `add_row_vector` benchmark
void run_add_vector_benchmark(
    const std::string &label,
    FlatMatrix<float> &A,
    const std::vector<float> &vec,
    std::size_t iterations = 10000,
    std::size_t warm_up_iters = 50)
{
    // Warm-up
    for (std::size_t i = 0; i < warm_up_iters; ++i)
    {
        A.add_row_vector(vec);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iterations; ++i)
    {
        A.add_row_vector(vec);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::nano> elapsed = (end - start) / iterations;
    printf("[%20s] Average Time: %.2f ns\n", label.c_str(), elapsed.count());

    // Force the compiler to care about the result
    // This tiny check makes it impossible to optimize away the multiplication
    if (A.rows() > 0 && A(0, 0) == 999.999f)
    {
        std::cout << "This will almost never happen, but the compiler doesn't know that.\n";
    }
}

void run_relu_benchmark(
    const std::string &label,
    FlatMatrix<float> &A,
    std::size_t iterations = 10000,
    std::size_t warm_up_iters = 50)
{
    // Warm-up
    for (std::size_t i = 0; i < warm_up_iters; ++i)
    {
        A.transform([](float x)
                    { return std::max(0.0f, x); });

        // Keep at least one value negative so the compiler cannot reason
        // that future ReLUs are always no-ops.
        // A(0, 0) = -1.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        A.transform([](float x)
                    { return std::max(0.0f, x); });

        // Prevent the benchmark from becoming "ReLU on already-ReLUed data"
        // in a completely trivial way.
        // A(0, 0) = -1.0f;
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::nano> elapsed = (end - start) / iterations;
    printf("[%20s] Average Time: %.2f ns\n", label.c_str(), elapsed.count());

    if (A.rows() > 0 && A(0, 0) == 999.999f)
    {
        std::cout << "This will almost never happen, but the compiler doesn't know that.\n";
    }
}

/* --- The Four Specific Benchmark Functions --- */

void benchmark_dense_by_dense(size_t M, size_t K, size_t N)
{
    auto A = MatrixGenerator::generate<float>(M, K, 0.0f, -1.0f, 1.0f);
    auto B = MatrixGenerator::generate<float>(N, K, 0.0f, -1.0f, 1.0f);
    run_matrix_mul_benchmark("Dense x Dense", A, B);
}

void benchmark_dense_by_sparse(size_t M, size_t K, size_t N)
{
    auto A = MatrixGenerator::generate<float>(M, K, 0.0f, -1.0f, 1.0f);
    auto B = MatrixGenerator::generate<float>(N, K, 0.95f, -1.0f, 1.0f); // 95% zeros
    run_matrix_mul_benchmark("Dense x Sparse", A, B);
}

void benchmark_sparse_by_dense(size_t M, size_t K, size_t N)
{
    auto A = MatrixGenerator::generate<float>(M, K, 0.95f, -1.0f, 1.0f); // 95% zeros
    auto B = MatrixGenerator::generate<float>(N, K, 0.0f, -1.0f, 1.0f);
    run_matrix_mul_benchmark("Sparse x Dense", A, B);
}

void benchmark_sparse_by_sparse(size_t M, size_t K, size_t N)
{
    auto A = MatrixGenerator::generate<float>(M, K, 0.95f, -1.0f, 1.0f); // 95% zeros
    auto B = MatrixGenerator::generate<float>(N, K, 0.95f, -1.0f, 1.0f); // 95% zeros
    run_matrix_mul_benchmark("Sparse x Sparse", A, B);
}

void benchmark_add_vector(std::size_t M, std::size_t K, std::size_t N)
{
    auto A = MatrixGenerator::generate<float>(M, K, 0.0f, -1.0f, 1.0f);
    auto Vec = MatrixGenerator::generateVec(K, 0.0f, 0.05f, 0.2f);
    run_add_vector_benchmark("Add vector", A, Vec);
}

void benchmark_relu(std::size_t M, std::size_t K, std::size_t N)
{
    (void)N;

    auto A = MatrixGenerator::generate<float>(M, K, 0.0f, -1.0f, 1.0f);
    run_relu_benchmark("Apply ReLU", A);
}

int main()
{
    // These dimensions simulate a typical small hidden layer
    const size_t M = 16;   // Batch size
    const size_t K = 4096; // Input features
    const size_t N = 4096; // Output features (transposed rows)

    std::cout << "Starting FlatMatrix Benchmarks...\n";
    std::cout << "Note: All should have similar runtimes because FlatMatrix is a Dense storage format.\n";
    std::cout << "----------------------------------------------------------\n";

    benchmark_dense_by_sparse(M, K, N);
    benchmark_sparse_by_dense(M, K, N);
    benchmark_sparse_by_sparse(M, K, N);
    benchmark_dense_by_dense(M, K, N);
    benchmark_add_vector(M, K, N);
    benchmark_relu(M, K, N);

    return 0;
}
