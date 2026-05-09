#ifndef FLAT_MATRIX_HPP
#define FLAT_MATRIX_HPP

#pragma once

#include <algorithm>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <cmath>
#include <format>
#include <map>

#include "MatrixConcepts.hpp"


template <typename T>
class FlatMatrix {
public:
    FlatMatrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols) {}

    // TODO: check if unnecessary copy is needed
    FlatMatrix(std::size_t rows, std::size_t cols, const std::vector<T>& data)
        : rows_(rows), cols_(cols), data_(data) {
        if (data.size() != rows * cols) {
            throw std::runtime_error("Data size does not match matrix dimensions.");
        }

        // std::cout << "FlatMatrix dimensions: " << rows_ << " x " << cols_ << "\n";
        // print_statistics();
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }

    const T& operator()(std::size_t r, std::size_t c) const {
        // TODO: experiment if removing bounds check improves performance
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix index out of range.");
        }
        return data_[r * cols_ + c];
    }

    __attribute__((noinline))
    void multiply_transposed_rhs(const FlatMatrix& rhs, FlatMatrix& out) const {
        // std::cout << "Multiplying matrix " << rows_ << " x " << cols_ << " by matrix " << rhs.cols() << " x " << rhs.rows() << "\n";
        if (cols_ != rhs.cols_) {
            throw std::runtime_error("Dimension mismatch.");
        }

        if (out.rows() != rows_ || out.cols() != rhs.rows()) {
            throw std::runtime_error("Output matrix has incorrect dimensions.");
        }

        // Required before passing pointers as __restrict__.
#if defined(FLATMATRIX_USE_DOT4)
        if (&out == this || &out == &rhs) {
            throw std::runtime_error("Output matrix must not alias input matrices.");
        }

        if constexpr (std::is_same_v<T, float>) {
            // multiply_transposed_rhs_float_dot4_kernel(
            multiply_transposed_rhs_float_dot4_kernel_4_4(
                data_.data(),
                rhs.data_.data(),
                out.data_.data(),
                rows_,
                rhs.rows_,
                cols_
            );
            return;
        }
#endif

        for (std::size_t i = 0; i < rows_; ++i) {
            const T* x = &data_[i * cols_];
            T* y = &out.data_[i * out.cols_];

            for (std::size_t j = 0; j < rhs.rows_; ++j) {
                const T* w = &rhs.data_[j * rhs.cols_];

                T sum = 0.0f;
                for (std::size_t k = 0; k < cols_; ++k) {
                    sum += x[k] * w[k];
                }

                y[j] = sum;
            }
        }
    }

    void add_row_vector(const std::vector<T>& vec) {
        if (vec.size() != cols_) {
            throw std::runtime_error("Vector size does not match number of columns.");
        }

        for (std::size_t r = 0; r < rows_; ++r) {
            T* row = &data_[r * cols_];
            for (std::size_t c = 0; c < cols_; ++c) {
                row[c] += vec[c];
            }
        }
    }

    template <typename Func>
    void transform(Func func) {
        for (T& val : data_) {
            val = func(val);
        }
    }

    std::vector<T> take_data() {
        // 1. Steal the data into a temporary return variable
        std::vector<T> stolen_data = std::move(data_);

        // 2. Mark this matrix as "empty" to prevent crashes
        rows_ = 0;
        cols_ = 0;

        // 3. Return the stolen data
        return stolen_data;
    }

    void print_statistics() const {
        if (data_.empty()) {
            std::cout << "Matrix is empty.\n";
            return;
        }

        size_t total_elements = data_.size();
        size_t non_zero_count = std::count_if(data_.begin(), data_.end(), [](T v) {
            return v != T{0};
        });

        double density = static_cast<double>(non_zero_count) / total_elements;
        double sparsity = 1.0 - density;

        // Descriptives
        auto [min_it, max_it] = std::minmax_element(data_.begin(), data_.end());
        T min_val = *min_it;
        T max_val = *max_it;

        double sum = std::accumulate(data_.begin(), data_.end(), 0.0);
        double mean = sum / total_elements;

        double sq_sum = std::inner_product(data_.begin(), data_.end(), data_.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / total_elements - mean * mean);

        // Header
        std::cout << "--- Matrix Statistics ---\n";
        std::cout << std::format("{:<15} : {} x {}\n", "Dimensions", rows_, cols_);
        std::cout << std::format("{:<15} : {:.2f}%\n", "Density", density * 100.0);
        std::cout << std::format("{:<15} : {:.2f}%\n", "Sparsity", sparsity * 100.0);

        // Value Stats
        std::cout << "\n--- Value Distribution ---\n";
        std::cout << std::format("{:<10} {:>10} {:>10} {:>10}\n", "Min", "Max", "Mean", "StdDev");
        std::cout << std::format("{:<10.4f} {:>10.4f} {:>10.4f} {:>10.4f}\n",
                                 (double)min_val, (double)max_val, mean, stdev);

        // Simple Histogram (5 Buckets)
        std::cout << "\n--- Range Spread ---\n";
        if (min_val != max_val) {
            const int BUCKETS = 5;
            std::vector<size_t> histogram(BUCKETS, 0);
            double range = static_cast<double>(max_val - min_val);

            for (T val : data_) {
                int bucket = std::min(static_cast<int>((val - min_val) / range * BUCKETS), BUCKETS - 1);
                histogram[bucket]++;
            }

            for (int i = 0; i < BUCKETS; ++i) {
                double lower = (double)min_val + (range / BUCKETS) * i;
                double upper = (double)min_val + (range / BUCKETS) * (i + 1);
                std::string bar(histogram[i] * 20 / total_elements, '#'); // Simple ASCII bar
                std::cout << std::format("[{:>7.2f}, {:>7.2f}]: {:<5} {}\n", lower, upper, histogram[i], bar);
            }
        } else {
            std::cout << "All values are identical.\n";
        }
        std::cout << "-------------------------\n";
    }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<T> data_;   // row-major storage

    __attribute__((noinline))
    static void multiply_transposed_rhs_float_dot4_kernel_2_4(
        const float* __restrict__ lhs,
        const float* __restrict__ rhs,
        float* __restrict__ out,
        std::size_t M,
        std::size_t N,
        std::size_t K
    ) {
        std::size_t i = 0;

        // Main 2x4 kernel:
        // computes 2 rows of out and 4 columns of out at once.
        for (; i + 2 <= M; i += 2) {
            const float* __restrict__ x0 = lhs + (i + 0) * K;
            const float* __restrict__ x1 = lhs + (i + 1) * K;

            float* __restrict__ y0 = out + (i + 0) * N;
            float* __restrict__ y1 = out + (i + 1) * N;

            std::size_t j = 0;

            for (; j + 4 <= N; j += 4) {
                const float* __restrict__ w0 = rhs + (j + 0) * K;
                const float* __restrict__ w1 = rhs + (j + 1) * K;
                const float* __restrict__ w2 = rhs + (j + 2) * K;
                const float* __restrict__ w3 = rhs + (j + 3) * K;

                float s00 = 0.0f;
                float s01 = 0.0f;
                float s02 = 0.0f;
                float s03 = 0.0f;

                float s10 = 0.0f;
                float s11 = 0.0f;
                float s12 = 0.0f;
                float s13 = 0.0f;

                for (std::size_t k = 0; k < K; ++k) {
                    const float a0 = x0[k];
                    const float a1 = x1[k];

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
                }

                y0[j + 0] = s00;
                y0[j + 1] = s01;
                y0[j + 2] = s02;
                y0[j + 3] = s03;

                y1[j + 0] = s10;
                y1[j + 1] = s11;
                y1[j + 2] = s12;
                y1[j + 3] = s13;
            }

            // Remainder columns for the 2-row block.
            for (; j < N; ++j) {
                const float* __restrict__ w = rhs + j * K;

                float s0 = 0.0f;
                float s1 = 0.0f;

                for (std::size_t k = 0; k < K; ++k) {
                    const float b = w[k];

                    s0 += x0[k] * b;
                    s1 += x1[k] * b;
                }

                y0[j] = s0;
                y1[j] = s1;
            }
        }

        // Remainder row if M is odd.
        for (; i < M; ++i) {
            const float* __restrict__ x = lhs + i * K;
            float* __restrict__ y = out + i * N;

            std::size_t j = 0;

            // Original 1x4 dot4 kernel for the last row.
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

            // Remainder columns for the last row.
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

    __attribute__((noinline))
    static void multiply_transposed_rhs_float_dot4_kernel_3_4(
        const float* __restrict__ lhs,
        const float* __restrict__ rhs,
        float* __restrict__ out,
        std::size_t M,
        std::size_t N,
        std::size_t K
    ) {
        std::size_t i = 0;

        // Main 3x4 kernel:
        // computes 3 rows of out and 4 columns of out at once.
        for (; i + 3 <= M; i += 3) {
            const float* __restrict__ x0 = lhs + (i + 0) * K;
            const float* __restrict__ x1 = lhs + (i + 1) * K;
            const float* __restrict__ x2 = lhs + (i + 2) * K;

            float* __restrict__ y0 = out + (i + 0) * N;
            float* __restrict__ y1 = out + (i + 1) * N;
            float* __restrict__ y2 = out + (i + 2) * N;

            std::size_t j = 0;

            for (; j + 4 <= N; j += 4) {
                const float* __restrict__ w0 = rhs + (j + 0) * K;
                const float* __restrict__ w1 = rhs + (j + 1) * K;
                const float* __restrict__ w2 = rhs + (j + 2) * K;
                const float* __restrict__ w3 = rhs + (j + 3) * K;

                float s00 = 0.0f, s01 = 0.0f, s02 = 0.0f, s03 = 0.0f;
                float s10 = 0.0f, s11 = 0.0f, s12 = 0.0f, s13 = 0.0f;
                float s20 = 0.0f, s21 = 0.0f, s22 = 0.0f, s23 = 0.0f;

                for (std::size_t k = 0; k < K; ++k) {
                    const float a0 = x0[k];
                    const float a1 = x1[k];
                    const float a2 = x2[k];

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
            }

            // Remainder columns for the 3-row block.
            for (; j < N; ++j) {
                const float* __restrict__ w = rhs + j * K;

                float s0 = 0.0f;
                float s1 = 0.0f;
                float s2 = 0.0f;

                for (std::size_t k = 0; k < K; ++k) {
                    const float b = w[k];

                    s0 += x0[k] * b;
                    s1 += x1[k] * b;
                    s2 += x2[k] * b;
                }

                y0[j] = s0;
                y1[j] = s1;
                y2[j] = s2;
            }
        }

        // Handle leftover rows with a 2x4 kernel.
        for (; i + 2 <= M; i += 2) {
            const float* __restrict__ x0 = lhs + (i + 0) * K;
            const float* __restrict__ x1 = lhs + (i + 1) * K;

            float* __restrict__ y0 = out + (i + 0) * N;
            float* __restrict__ y1 = out + (i + 1) * N;

            std::size_t j = 0;

            for (; j + 4 <= N; j += 4) {
                const float* __restrict__ w0 = rhs + (j + 0) * K;
                const float* __restrict__ w1 = rhs + (j + 1) * K;
                const float* __restrict__ w2 = rhs + (j + 2) * K;
                const float* __restrict__ w3 = rhs + (j + 3) * K;

                float s00 = 0.0f, s01 = 0.0f, s02 = 0.0f, s03 = 0.0f;
                float s10 = 0.0f, s11 = 0.0f, s12 = 0.0f, s13 = 0.0f;

                for (std::size_t k = 0; k < K; ++k) {
                    const float a0 = x0[k];
                    const float a1 = x1[k];

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
                }

                y0[j + 0] = s00;
                y0[j + 1] = s01;
                y0[j + 2] = s02;
                y0[j + 3] = s03;

                y1[j + 0] = s10;
                y1[j + 1] = s11;
                y1[j + 2] = s12;
                y1[j + 3] = s13;
            }

            // Remainder columns for the 2-row block.
            for (; j < N; ++j) {
                const float* __restrict__ w = rhs + j * K;

                float s0 = 0.0f;
                float s1 = 0.0f;

                for (std::size_t k = 0; k < K; ++k) {
                    const float b = w[k];

                    s0 += x0[k] * b;
                    s1 += x1[k] * b;
                }

                y0[j] = s0;
                y1[j] = s1;
            }
        }

        // Handle final leftover row with the original 1x4 kernel.
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

            // Remainder columns for the last row.
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


    __attribute__((noinline))
    static void multiply_transposed_rhs_float_dot4_kernel_4_4(
        const float* __restrict__ lhs,
        const float* __restrict__ rhs,
        float* __restrict__ out,
        std::size_t M,
        std::size_t N,
        std::size_t K
    ) {
        std::size_t i = 0;

        // Main 4x4 kernel:
        // computes 4 rows of out and 4 columns of out at once.
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

            // Remainder columns for the 4-row block.
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

        // Handle leftover rows with the original 1x4 kernel.
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

            // Remainder columns for the final leftover rows.
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


    __attribute__((noinline))
    static void multiply_transposed_rhs_float_dot4_kernel_6_4(
        const float* __restrict__ lhs,
        const float* __restrict__ rhs,
        float* __restrict__ out,
        std::size_t M,
        std::size_t N,
        std::size_t K
    ) {
        std::size_t i = 0;

        // Main 6x4 kernel:
        // computes 6 rows of out and 4 columns of out at once.
        for (; i + 6 <= M; i += 6) {
            const float* __restrict__ x0 = lhs + (i + 0) * K;
            const float* __restrict__ x1 = lhs + (i + 1) * K;
            const float* __restrict__ x2 = lhs + (i + 2) * K;
            const float* __restrict__ x3 = lhs + (i + 3) * K;
            const float* __restrict__ x4 = lhs + (i + 4) * K;
            const float* __restrict__ x5 = lhs + (i + 5) * K;

            float* __restrict__ y0 = out + (i + 0) * N;
            float* __restrict__ y1 = out + (i + 1) * N;
            float* __restrict__ y2 = out + (i + 2) * N;
            float* __restrict__ y3 = out + (i + 3) * N;
            float* __restrict__ y4 = out + (i + 4) * N;
            float* __restrict__ y5 = out + (i + 5) * N;

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
                float s40 = 0.0f, s41 = 0.0f, s42 = 0.0f, s43 = 0.0f;
                float s50 = 0.0f, s51 = 0.0f, s52 = 0.0f, s53 = 0.0f;

                for (std::size_t k = 0; k < K; ++k) {
                    const float a0 = x0[k];
                    const float a1 = x1[k];
                    const float a2 = x2[k];
                    const float a3 = x3[k];
                    const float a4 = x4[k];
                    const float a5 = x5[k];

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

                    s40 += a4 * b0;
                    s41 += a4 * b1;
                    s42 += a4 * b2;
                    s43 += a4 * b3;

                    s50 += a5 * b0;
                    s51 += a5 * b1;
                    s52 += a5 * b2;
                    s53 += a5 * b3;
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

                y4[j + 0] = s40;
                y4[j + 1] = s41;
                y4[j + 2] = s42;
                y4[j + 3] = s43;

                y5[j + 0] = s50;
                y5[j + 1] = s51;
                y5[j + 2] = s52;
                y5[j + 3] = s53;
            }

            // Remainder columns for the 6-row block.
            for (; j < N; ++j) {
                const float* __restrict__ w = rhs + j * K;

                float s0 = 0.0f;
                float s1 = 0.0f;
                float s2 = 0.0f;
                float s3 = 0.0f;
                float s4 = 0.0f;
                float s5 = 0.0f;

                for (std::size_t k = 0; k < K; ++k) {
                    const float b = w[k];

                    s0 += x0[k] * b;
                    s1 += x1[k] * b;
                    s2 += x2[k] * b;
                    s3 += x3[k] * b;
                    s4 += x4[k] * b;
                    s5 += x5[k] * b;
                }

                y0[j] = s0;
                y1[j] = s1;
                y2[j] = s2;
                y3[j] = s3;
                y4[j] = s4;
                y5[j] = s5;
            }
        }

        // Handle leftover rows with a simple 1x4 kernel.
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

            // Remainder columns for the final leftover rows.
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
};


static_assert(IMatrix<FlatMatrix<float>, float>, 
    "FlatMatrix does not perfectly implement the IMatrix concept!");

#endif // FLAT_MATRIX_HPP
