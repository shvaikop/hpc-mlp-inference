#ifndef MATRIX_ALGORITHMS_HPP
#define MATRIX_ALGORITHMS_HPP

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include "MatrixConcepts.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define FLATMATRIX_NOINLINE __attribute__((noinline))
#else
#define FLATMATRIX_NOINLINE
#endif

namespace flatmatrix_detail {

inline bool byte_ranges_overlap(
    const void* a,
    std::size_t a_bytes,
    const void* b,
    std::size_t b_bytes
) noexcept {
    if (a == nullptr || b == nullptr || a_bytes == 0 || b_bytes == 0) {
        return false;
    }

    const auto a_begin = reinterpret_cast<std::uintptr_t>(a);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(b);

    const auto a_end = a_begin + a_bytes;
    const auto b_end = b_begin + b_bytes;

    return a_begin < b_end && b_begin < a_end;
}

FLATMATRIX_NOINLINE
static void multiply_transposed_rhs_float_dot4_kernel_4_4(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    std::size_t M,
    std::size_t N,
    std::size_t K
) {
    std::size_t i = 0;

    for (; i + 4 <= M; i += 4) {
        const float* __restrict__ x0 = lhs + (i + 0) * K;
        const float* __restrict__ x1 = lhs + (i + 1) * K;
        const float* __restrict__ x2 = lhs + (i + 2) * K;
        const float* __restrict__ x3 = lhs + (i + 3) * K;

        float* __restrict__ y0 = out + (i + 0) * N;
        float* __restrict__ y1 = out + (i + 1) * N;
        float* __restrict__ y2 = out + (i + 2) * N;
        float* __restrict__ y3 = out + (i + 3) * N;

        std::size_t j = 0;

        for (; j + 4 <= N; j += 4) {
            const float* __restrict__ w0 = rhs + (j + 0) * K;
            const float* __restrict__ w1 = rhs + (j + 1) * K;
            const float* __restrict__ w2 = rhs + (j + 2) * K;
            const float* __restrict__ w3 = rhs + (j + 3) * K;

            float s00 = 0.0f, s01 = 0.0f, s02 = 0.0f, s03 = 0.0f;
            float s10 = 0.0f, s11 = 0.0f, s12 = 0.0f, s13 = 0.0f;
            float s20 = 0.0f, s21 = 0.0f, s22 = 0.0f, s23 = 0.0f;
            float s30 = 0.0f, s31 = 0.0f, s32 = 0.0f, s33 = 0.0f;

            for (std::size_t k = 0; k < K; ++k) {
                const float a0 = x0[k];
                const float a1 = x1[k];
                const float a2 = x2[k];
                const float a3 = x3[k];

                const float b0 = w0[k];
                const float b1 = w1[k];
                const float b2 = w2[k];
                const float b3 = w3[k];

                s00 += a0 * b0;
                s01 += a0 * b1;
                s02 += a0 * b2;
                s03 += a0 * b3;

                s10 += a1 * b0;
                s11 += a1 * b1;
                s12 += a1 * b2;
                s13 += a1 * b3;

                s20 += a2 * b0;
                s21 += a2 * b1;
                s22 += a2 * b2;
                s23 += a2 * b3;

                s30 += a3 * b0;
                s31 += a3 * b1;
                s32 += a3 * b2;
                s33 += a3 * b3;
            }

            y0[j + 0] = s00;
            y0[j + 1] = s01;
            y0[j + 2] = s02;
            y0[j + 3] = s03;

            y1[j + 0] = s10;
            y1[j + 1] = s11;
            y1[j + 2] = s12;
            y1[j + 3] = s13;

            y2[j + 0] = s20;
            y2[j + 1] = s21;
            y2[j + 2] = s22;
            y2[j + 3] = s23;

            y3[j + 0] = s30;
            y3[j + 1] = s31;
            y3[j + 2] = s32;
            y3[j + 3] = s33;
        }

        for (; j < N; ++j) {
            const float* __restrict__ w = rhs + j * K;

            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;

            for (std::size_t k = 0; k < K; ++k) {
                const float b = w[k];

                s0 += x0[k] * b;
                s1 += x1[k] * b;
                s2 += x2[k] * b;
                s3 += x3[k] * b;
            }

            y0[j] = s0;
            y1[j] = s1;
            y2[j] = s2;
            y3[j] = s3;
        }
    }

    for (; i < M; ++i) {
        const float* __restrict__ x = lhs + i * K;
        float* __restrict__ y = out + i * N;

        std::size_t j = 0;

        for (; j + 4 <= N; j += 4) {
            const float* __restrict__ w0 = rhs + (j + 0) * K;
            const float* __restrict__ w1 = rhs + (j + 1) * K;
            const float* __restrict__ w2 = rhs + (j + 2) * K;
            const float* __restrict__ w3 = rhs + (j + 3) * K;

            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;

            for (std::size_t k = 0; k < K; ++k) {
                const float a = x[k];

                s0 += a * w0[k];
                s1 += a * w1[k];
                s2 += a * w2[k];
                s3 += a * w3[k];
            }

            y[j + 0] = s0;
            y[j + 1] = s1;
            y[j + 2] = s2;
            y[j + 3] = s3;
        }

        for (; j < N; ++j) {
            const float* __restrict__ w = rhs + j * K;

            float sum = 0.0f;

            for (std::size_t k = 0; k < K; ++k) {
                sum += x[k] * w[k];
            }

            y[j] = sum;
        }
    }
}

#if defined(__AVX2__)
FLATMATRIX_NOINLINE
static void add_row_vector_avx2_impl(
    float* __restrict__ data,
    const float* __restrict__ vec,
    std::size_t rows,
    std::size_t cols
) {
    const std::size_t vec_end = cols & ~std::size_t(7);

    for (std::size_t r = 0; r < rows; ++r) {
        float* __restrict__ row = data + r * cols;

        std::size_t c = 0;

        for (; c < vec_end; c += 8) {
            __m256 v = _mm256_loadu_ps(vec + c);
            __m256 x = _mm256_loadu_ps(row + c);
            x = _mm256_add_ps(x, v);
            _mm256_storeu_ps(row + c, x);
        }

        switch (cols - vec_end) {
            case 7: row[c + 6] += vec[c + 6]; [[fallthrough]];
            case 6: row[c + 5] += vec[c + 5]; [[fallthrough]];
            case 5: row[c + 4] += vec[c + 4]; [[fallthrough]];
            case 4: row[c + 3] += vec[c + 3]; [[fallthrough]];
            case 3: row[c + 2] += vec[c + 2]; [[fallthrough]];
            case 2: row[c + 1] += vec[c + 1]; [[fallthrough]];
            case 1: row[c + 0] += vec[c + 0]; [[fallthrough]];
            case 0: break;
        }
    }
}
#endif

#if defined(__AVX__)
FLATMATRIX_NOINLINE
static void relu_inplace_avx_impl(float* data, std::size_t n) {
    const __m256 zero = _mm256_setzero_ps();

    std::size_t i = 0;
    const std::size_t vec_end = n & ~std::size_t(7);

    for (; i < vec_end; i += 8) {
        __m256 x = _mm256_loadu_ps(data + i);
        x = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(data + i, x);
    }

    for (; i < n; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}
#endif

} // namespace flatmatrix_detail

template <MatrixLike Lhs, MatrixLike Rhs, WritableMatrix Out>
FLATMATRIX_NOINLINE
void multiply_transposed_rhs(const Lhs& lhs, const Rhs& rhs, Out& out) {
    using LhsT = matrix_scalar_t<Lhs>;
    using RhsT = matrix_scalar_t<Rhs>;
    using OutT = matrix_scalar_t<Out>;

    if (lhs.cols() != rhs.cols()) {
        throw std::runtime_error("Dimension mismatch: lhs.cols() must equal rhs.cols().");
    }

    if (out.rows() != lhs.rows() || out.cols() != rhs.rows()) {
        throw std::runtime_error("Output matrix has incorrect dimensions.");
    }

#if defined(FLATMATRIX_USE_DOT4)
    if constexpr (
        std::is_same_v<LhsT, float> &&
        std::is_same_v<RhsT, float> &&
        std::is_same_v<OutT, float>
    ) {
        const float* lhs_data = lhs.data();
        const float* rhs_data = rhs.data();
        float* out_data = out.data();

        const std::size_t lhs_bytes = lhs.rows() * lhs.cols() * sizeof(float);
        const std::size_t rhs_bytes = rhs.rows() * rhs.cols() * sizeof(float);
        const std::size_t out_bytes = out.rows() * out.cols() * sizeof(float);

        if (
            flatmatrix_detail::byte_ranges_overlap(out_data, out_bytes, lhs_data, lhs_bytes) ||
            flatmatrix_detail::byte_ranges_overlap(out_data, out_bytes, rhs_data, rhs_bytes)
        ) {
            throw std::runtime_error("Output matrix must not alias input matrices.");
        }

        flatmatrix_detail::multiply_transposed_rhs_float_dot4_kernel_4_4(
            lhs_data,
            rhs_data,
            out_data,
            lhs.rows(),
            rhs.rows(),
            lhs.cols()
        );

        return;
    }
#endif

    for (std::size_t i = 0; i < lhs.rows(); ++i) {
        const auto* x = lhs.data() + i * lhs.cols();
        auto* y = out.data() + i * out.cols();

        for (std::size_t j = 0; j < rhs.rows(); ++j) {
            const auto* w = rhs.data() + j * rhs.cols();

            OutT sum{};

            for (std::size_t k = 0; k < lhs.cols(); ++k) {
                sum += static_cast<OutT>(x[k] * w[k]);
            }

            y[j] = sum;
        }
    }
}

template <WritableMatrix Matrix>
FLATMATRIX_NOINLINE
void add_row_vector(Matrix& matrix, const std::vector<matrix_scalar_t<Matrix>>& vec) {
    using T = matrix_scalar_t<Matrix>;

    if (vec.size() != matrix.cols()) {
        throw std::runtime_error("Vector size does not match number of columns.");
    }

#if defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
        flatmatrix_detail::add_row_vector_avx2_impl(
            matrix.data(),
            vec.data(),
            matrix.rows(),
            matrix.cols()
        );
        return;
    }
#endif

    for (std::size_t r = 0; r < matrix.rows(); ++r) {
        T* row = matrix.data() + r * matrix.cols();

        for (std::size_t c = 0; c < matrix.cols(); ++c) {
            row[c] += vec[c];
        }
    }
}

template <WritableMatrix Matrix, typename Func>
void transform(Matrix& matrix, Func func) {
    auto* data = matrix.data();

    for (std::size_t i = 0; i < matrix.size(); ++i) {
        data[i] = func(data[i]);
    }
}

template <WritableMatrix Matrix>
FLATMATRIX_NOINLINE
void apply_relu(Matrix& matrix) {
    using T = matrix_scalar_t<Matrix>;

#if defined(__AVX__)
    if constexpr (std::is_same_v<T, float>) {
        flatmatrix_detail::relu_inplace_avx_impl(matrix.data(), matrix.size());
        return;
    }
#endif

    auto* data = matrix.data();

    for (std::size_t i = 0; i < matrix.size(); ++i) {
        data[i] = std::max(T{0}, data[i]);
    }
}

#undef FLATMATRIX_NOINLINE

#endif // MATRIX_ALGORITHMS_HPP
