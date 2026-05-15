#include <algorithm>
#include <chrono>
#include <format>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "FlatMatrix.hpp"
#include "FlatMatrixView.hpp"
#include "MatrixAlgorithms.hpp"

namespace MatrixGenerator {

template <typename T>
FlatMatrix<T> generate(
    std::size_t rows,
    std::size_t cols,
    float sparsity,
    T min_val,
    T max_val
) {
    std::vector<T> data;
    data.reserve(rows * cols);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<T> val_dist(min_val, max_val);
    std::bernoulli_distribution zero_dist(sparsity);

    for (std::size_t i = 0; i < rows * cols; ++i) {
        if (zero_dist(gen)) {
            data.push_back(T{0});
        } else {
            data.push_back(val_dist(gen));
        }
    }

    return FlatMatrix<T>(rows, cols, std::move(data));
}

template <typename T>
std::vector<T> generateVec(
    std::size_t size,
    float sparsity,
    T min_val,
    T max_val
) {
    std::vector<T> data;
    data.reserve(size);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<T> val_dist(min_val, max_val);
    std::bernoulli_distribution zero_dist(sparsity);

    for (std::size_t i = 0; i < size; ++i) {
        if (zero_dist(gen)) {
            data.push_back(T{0});
        } else {
            data.push_back(val_dist(gen));
        }
    }

    return data;
}

} // namespace MatrixGenerator

static void prevent_optimize_away(const FlatMatrix<float>& C)
{
    if (C.rows() > 0 && C.cols() > 0 && C(0, 0) == 999.999f) {
        std::cout << "This will almost never happen.\n";
    }
}

static void prevent_optimize_away_view(FlatMatrixView<float> C)
{
    if (C.rows() > 0 && C.cols() > 0 && C(0, 0) == 999.999f) {
        std::cout << "This will almost never happen.\n";
    }
}

void run_matrix_mul_benchmark_flatmatrix(
    const std::string& label,
    const FlatMatrix<float>& A,
    const FlatMatrix<float>& B,
    int iterations = 20,
    std::size_t warm_up_iters = 5
) {
    FlatMatrix<float> C(A.rows(), B.rows());

    for (std::size_t i = 0; i < warm_up_iters; ++i) {
        multiply_transposed_rhs(A, B, C);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        multiply_transposed_rhs(A, B, C);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed =
        (end - start) / iterations;

    std::cout << std::format(
        "[{:^28}] Average Time: {:.3f} ms\n",
        label,
        elapsed.count()
    );

    prevent_optimize_away(C);
}

void run_matrix_mul_benchmark_view(
    const std::string& label,
    const FlatMatrix<float>& A_storage,
    const FlatMatrix<float>& B_storage,
    int iterations = 20,
    std::size_t warm_up_iters = 5
) {
    FlatMatrix<float> C_storage(A_storage.rows(), B_storage.rows());

    FlatMatrixView<const float> A = A_storage.view();
    FlatMatrixView<const float> B = B_storage.view();
    FlatMatrixView<float> C = C_storage.view();

    for (std::size_t i = 0; i < warm_up_iters; ++i) {
        multiply_transposed_rhs(A, B, C);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        multiply_transposed_rhs(A, B, C);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed =
        (end - start) / iterations;

    std::cout << std::format(
        "[{:^28}] Average Time: {:.3f} ms\n",
        label,
        elapsed.count()
    );

    prevent_optimize_away_view(C);
}

void run_add_vector_benchmark_flatmatrix(
    const std::string& label,
    FlatMatrix<float>& A,
    const std::vector<float>& vec,
    std::size_t iterations = 10000,
    std::size_t warm_up_iters = 50
) {
    for (std::size_t i = 0; i < warm_up_iters; ++i) {
        add_row_vector(A, vec);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < iterations; ++i) {
        add_row_vector(A, vec);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::nano> elapsed =
        (end - start) / iterations;

    std::cout << std::format(
        "[{:^28}] Average Time: {:.2f} ns\n",
        label,
        elapsed.count()
    );

    prevent_optimize_away(A);
}

void run_add_vector_benchmark_view(
    const std::string& label,
    FlatMatrix<float>& A_storage,
    const std::vector<float>& vec,
    std::size_t iterations = 10000,
    std::size_t warm_up_iters = 50
) {
    FlatMatrixView<float> A = A_storage.view();

    for (std::size_t i = 0; i < warm_up_iters; ++i) {
        add_row_vector(A, vec);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < iterations; ++i) {
        add_row_vector(A, vec);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::nano> elapsed =
        (end - start) / iterations;

    std::cout << std::format(
        "[{:^28}] Average Time: {:.2f} ns\n",
        label,
        elapsed.count()
    );

    prevent_optimize_away_view(A);
}

void run_relu_benchmark_flatmatrix(
    const std::string& label,
    FlatMatrix<float>& A,
    std::size_t iterations = 10000,
    std::size_t warm_up_iters = 50
) {
    for (std::size_t i = 0; i < warm_up_iters; ++i) {
        apply_relu(A);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < iterations; ++i) {
        apply_relu(A);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::nano> elapsed =
        (end - start) / iterations;

    std::cout << std::format(
        "[{:^28}] Average Time: {:.2f} ns\n",
        label,
        elapsed.count()
    );

    prevent_optimize_away(A);
}

void run_relu_benchmark_view(
    const std::string& label,
    FlatMatrix<float>& A_storage,
    std::size_t iterations = 10000,
    std::size_t warm_up_iters = 50
) {
    FlatMatrixView<float> A = A_storage.view();

    for (std::size_t i = 0; i < warm_up_iters; ++i) {
        apply_relu(A);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < iterations; ++i) {
        apply_relu(A);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::nano> elapsed =
        (end - start) / iterations;

    std::cout << std::format(
        "[{:^28}] Average Time: {:.2f} ns\n",
        label,
        elapsed.count()
    );

    prevent_optimize_away_view(A);
}

void benchmark_matrix_mul_pair(
    const std::string& label,
    std::size_t M,
    std::size_t K,
    std::size_t N,
    float sparsity_A,
    float sparsity_B
) {
    auto A = MatrixGenerator::generate<float>(M, K, sparsity_A, -1.0f, 1.0f);
    auto B = MatrixGenerator::generate<float>(N, K, sparsity_B, -1.0f, 1.0f);

    run_matrix_mul_benchmark_flatmatrix(label + " / FlatMatrix", A, B);
    run_matrix_mul_benchmark_view(label + " / View", A, B);
}

void benchmark_add_vector(std::size_t M, std::size_t K)
{
    auto A1 = MatrixGenerator::generate<float>(M, K, 0.0f, -1.0f, 1.0f);
    auto A2 = A1;

    auto vec = MatrixGenerator::generateVec<float>(K, 0.0f, 0.05f, 0.2f);

    run_add_vector_benchmark_view("Add vector / View", A2, vec);
    run_add_vector_benchmark_flatmatrix("Add vector / FlatMatrix", A1, vec);
}

void benchmark_relu(std::size_t M, std::size_t K)
{
    auto A1 = MatrixGenerator::generate<float>(M, K, 0.0f, -1.0f, 1.0f);
    auto A2 = A1;

    run_relu_benchmark_view("Apply ReLU / View", A2);
    run_relu_benchmark_flatmatrix("Apply ReLU / FlatMatrix", A1);
}

int main()
{
    const std::size_t M = 16;
    const std::size_t K = 4096;
    const std::size_t N = 4096;

    std::cout << "Starting FlatMatrix / FlatMatrixView Benchmarks...\n";
    std::cout << "M = " << M << ", K = " << K << ", N = " << N << '\n';
    std::cout << "----------------------------------------------------------\n";

    benchmark_matrix_mul_pair(
        "Dense x Sparse",
        M,
        K,
        N,
        0.0f,
        0.95f
    );

    benchmark_matrix_mul_pair(
        "Sparse x Dense",
        M,
        K,
        N,
        0.95f,
        0.0f
    );

    benchmark_matrix_mul_pair(
        "Sparse x Sparse",
        M,
        K,
        N,
        0.95f,
        0.95f
    );

    benchmark_matrix_mul_pair(
        "Dense x Dense",
        M,
        K,
        N,
        0.0f,
        0.0f
    );

    benchmark_add_vector(M, K);
    benchmark_relu(M, K);

    return 0;
}