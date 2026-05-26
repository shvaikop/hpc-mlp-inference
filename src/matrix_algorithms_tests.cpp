#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "FlatMatrix.hpp"
#include "FlatMatrixView.hpp"
#include "MatrixAlgorithms.hpp"

namespace TestData {

template <typename T>
FlatMatrix<T> make_matrix(std::size_t rows, std::size_t cols, T start = T{1}) {
    std::vector<T> data;
    data.reserve(rows * cols);

    for (std::size_t i = 0; i < rows * cols; ++i) {
        // Small deterministic values, including negatives.
        const int v = static_cast<int>((i + static_cast<std::size_t>(start)) % 11) - 5;
        data.push_back(static_cast<T>(v) / static_cast<T>(3));
    }

    return FlatMatrix<T>(rows, cols, std::move(data));
}

template <typename T>
FlatMatrix<T> make_random_matrix(
    std::size_t rows,
    std::size_t cols,
    std::uint32_t seed,
    T min_value = T{-1},
    T max_value = T{1}
) {
    std::vector<T> data;
    data.reserve(rows * cols);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(min_value, max_value);

    for (std::size_t i = 0; i < rows * cols; ++i) {
        data.push_back(dist(rng));
    }

    return FlatMatrix<T>(rows, cols, std::move(data));
}

template <typename T>
std::vector<T> make_vector(std::size_t size, T start = T{1}) {
    std::vector<T> data;
    data.reserve(size);

    for (std::size_t i = 0; i < size; ++i) {
        const int v = static_cast<int>((i + static_cast<std::size_t>(start)) % 7) - 3;
        data.push_back(static_cast<T>(v) / static_cast<T>(5));
    }

    return data;
}

} // namespace TestData

namespace TestUtil {

void fail(const std::string& message) {
    std::cerr << "[FAIL] " << message << '\n';
    std::exit(EXIT_FAILURE);
}

void pass(const std::string& message) {
    std::cout << "[PASS] " << message << '\n';
}

bool nearly_equal(float a, float b, float abs_eps = 1e-4f, float rel_eps = 1e-4f) {
    const float diff = std::fabs(a - b);
    if (diff <= abs_eps) {
        return true;
    }
    return diff <= rel_eps * std::max(std::fabs(a), std::fabs(b));
}

template <typename MatrixLikeA, typename MatrixLikeB>
void assert_matrix_near(
    const MatrixLikeA& actual,
    const MatrixLikeB& expected,
    const std::string& label,
    float abs_eps = 1e-4f,
    float rel_eps = 1e-4f
) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
        fail(label + ": shape mismatch");
    }

    for (std::size_t r = 0; r < actual.rows(); ++r) {
        for (std::size_t c = 0; c < actual.cols(); ++c) {
            const float a = actual(r, c);
            const float e = expected(r, c);
            if (!nearly_equal(a, e, abs_eps, rel_eps)) {
                std::cerr << std::setprecision(9)
                          << "Mismatch in " << label
                          << " at (" << r << ", " << c << "): actual=" << a
                          << ", expected=" << e << '\n';
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

template <typename MatrixLike>
long double checksum(const MatrixLike& matrix) {
    // Weighted checksum reduces the chance of accidentally missing permutations.
    long double sum = 0.0L;
    for (std::size_t r = 0; r < matrix.rows(); ++r) {
        for (std::size_t c = 0; c < matrix.cols(); ++c) {
            const long double weight = 1.0L + static_cast<long double>((r * 131 + c * 17) % 97);
            sum += static_cast<long double>(matrix(r, c)) * weight;
        }
    }
    return sum;
}

bool checksum_nearly_equal(long double a, long double b, long double eps = 1e-3L) {
    const long double diff = std::fabs(a - b);
    if (diff <= eps) {
        return true;
    }
    return diff <= eps * std::max(std::fabs(a), std::fabs(b));
}

} // namespace TestUtil

namespace Reference {

// Computes C = A * B^T, matching multiply_transposed_rhs(A, B, C),
// where A has shape M x K and B has shape N x K.
template <typename MatrixLikeA, typename MatrixLikeB>
FlatMatrix<float> multiply_transposed_rhs(const MatrixLikeA& A, const MatrixLikeB& B) {
    if (A.cols() != B.cols()) {
        TestUtil::fail("Reference multiply_transposed_rhs: incompatible shapes");
    }

    FlatMatrix<float> C(A.rows(), B.rows());

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < B.rows(); ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * B(j, k);
            }
            C(i, j) = sum;
        }
    }

    return C;
}

template <typename MatrixLike>
FlatMatrix<float> add_row_vector(const MatrixLike& A, const std::vector<float>& vec) {
    if (A.cols() != vec.size()) {
        TestUtil::fail("Reference add_row_vector: incompatible shapes");
    }

    FlatMatrix<float> out(A.rows(), A.cols());
    for (std::size_t r = 0; r < A.rows(); ++r) {
        for (std::size_t c = 0; c < A.cols(); ++c) {
            out(r, c) = A(r, c) + vec[c];
        }
    }
    return out;
}

template <typename MatrixLike>
FlatMatrix<float> relu(const MatrixLike& A) {
    FlatMatrix<float> out(A.rows(), A.cols());
    for (std::size_t r = 0; r < A.rows(); ++r) {
        for (std::size_t c = 0; c < A.cols(); ++c) {
            out(r, c) = std::max(0.0f, A(r, c));
        }
    }
    return out;
}

} // namespace Reference

void test_small_matrix_multiply_flatmatrix() {
    auto A = TestData::make_matrix<float>(3, 5, 1.0f);
    auto B = TestData::make_matrix<float>(4, 5, 2.0f); // B is stored as N x K.

    FlatMatrix<float> C(A.rows(), B.rows());
    multiply_transposed_rhs(A, B, C);

    auto expected = Reference::multiply_transposed_rhs(A, B);
    TestUtil::assert_matrix_near(C, expected, "small multiply / FlatMatrix");
    TestUtil::pass("small multiply / FlatMatrix");
}

void test_small_matrix_multiply_view() {
    auto A_storage = TestData::make_matrix<float>(3, 5, 3.0f);
    auto B_storage = TestData::make_matrix<float>(4, 5, 4.0f); // B is stored as N x K.
    FlatMatrix<float> C_storage(A_storage.rows(), B_storage.rows());

    const FlatMatrix<float>& A_const = A_storage;
    const FlatMatrix<float>& B_const = B_storage;

    auto A = A_const.view();  // should return FlatMatrixView<const float>
    auto B = B_const.view();  // should return FlatMatrixView<const float>
    auto C = C_storage.view();

    multiply_transposed_rhs(A, B, C);

    auto expected = Reference::multiply_transposed_rhs(A_storage, B_storage);
    TestUtil::assert_matrix_near(C_storage, expected, "small multiply / View");
    TestUtil::pass("small multiply / View");
}

void test_add_row_vector_flatmatrix() {
    auto A = TestData::make_matrix<float>(4, 6, 5.0f);
    const auto vec = TestData::make_vector<float>(A.cols(), 2.0f);

    auto expected = Reference::add_row_vector(A, vec);
    add_row_vector(A, vec);

    TestUtil::assert_matrix_near(A, expected, "add row vector / FlatMatrix");
    TestUtil::pass("add row vector / FlatMatrix");
}

void test_add_row_vector_view() {
    auto A_storage = TestData::make_matrix<float>(4, 6, 6.0f);
    const auto vec = TestData::make_vector<float>(A_storage.cols(), 3.0f);

    auto expected = Reference::add_row_vector(A_storage, vec);
    FlatMatrixView<float> A = A_storage.view();
    add_row_vector(A, vec);

    TestUtil::assert_matrix_near(A_storage, expected, "add row vector / View");
    TestUtil::pass("add row vector / View");
}

void test_relu_flatmatrix() {
    auto A = TestData::make_matrix<float>(5, 7, 7.0f);
    auto expected = Reference::relu(A);

    apply_relu(A);

    TestUtil::assert_matrix_near(A, expected, "ReLU / FlatMatrix");
    TestUtil::pass("ReLU / FlatMatrix");
}

void test_relu_view() {
    auto A_storage = TestData::make_matrix<float>(5, 7, 8.0f);
    auto expected = Reference::relu(A_storage);

    FlatMatrixView<float> A = A_storage.view();
    apply_relu(A);

    TestUtil::assert_matrix_near(A_storage, expected, "ReLU / View");
    TestUtil::pass("ReLU / View");
}

void stress_test_large_matrix_multiply_checksum() {
    constexpr std::size_t M = 16;
    constexpr std::size_t K = 4096;
    constexpr std::size_t N = 4096;
    constexpr int iterations = 20;

    auto A_storage = TestData::make_random_matrix<float>(M, K, 12345, -1.0f, 1.0f);
    auto B_storage = TestData::make_random_matrix<float>(N, K, 67890, -1.0f, 1.0f);
    FlatMatrix<float> C_storage(M, N);

    const auto& A_const = A_storage;
    const auto& B_const = B_storage;

    auto A = A_const.view();
    auto B = B_const.view();
    auto C = C_storage.view();

    long double reference_checksum = 0.0L;

    for (int iter = 0; iter < iterations; ++iter) {
        multiply_transposed_rhs(A, B, C);
        const long double current_checksum = TestUtil::checksum(C_storage);

        if (iter == 0) {
            reference_checksum = current_checksum;
        } else if (!TestUtil::checksum_nearly_equal(current_checksum, reference_checksum)) {
            std::cerr << std::setprecision(18)
                      << "Large matrix stress test failed at iteration " << iter
                      << ": checksum=" << current_checksum
                      << ", reference=" << reference_checksum << '\n';
            std::exit(EXIT_FAILURE);
        }
    }

    std::cout << std::setprecision(18)
              << "[PASS] large multiply stress / checksum = "
              << reference_checksum << '\n';
}

int main() {
    std::cout << "Starting MatrixAlgorithms tests...\n";
    std::cout << "----------------------------------\n";

    test_small_matrix_multiply_flatmatrix();
    test_small_matrix_multiply_view();
    test_add_row_vector_flatmatrix();
    test_add_row_vector_view();
    test_relu_flatmatrix();
    test_relu_view();
    stress_test_large_matrix_multiply_checksum();

    std::cout << "----------------------------------\n";
    std::cout << "All tests passed.\n";
    return EXIT_SUCCESS;
}
