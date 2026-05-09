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
            multiply_transposed_rhs_float_dot4_kernel(
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
    static void multiply_transposed_rhs_float_dot4_kernel(
        const float* __restrict__ lhs,
        const float* __restrict__ rhs,
        float* __restrict__ out,
        std::size_t M,
        std::size_t N,
        std::size_t K
    ) {
        for (std::size_t i = 0; i < M; ++i) {
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

                #pragma omp simd reduction(+:s0,s1,s2,s3)
                for (std::size_t k = 0; k < K; ++k) {
                    const float xv = x[k];

                    s0 += xv * w0[k];
                    s1 += xv * w1[k];
                    s2 += xv * w2[k];
                    s3 += xv * w3[k];
                }

                y[j + 0] = s0;
                y[j + 1] = s1;
                y[j + 2] = s2;
                y[j + 3] = s3;
            }

            // Remainder columns.
            for (; j < N; ++j) {
                const float* __restrict__ w = rhs + j * K;

                float sum = 0.0f;

                #pragma omp simd reduction(+:sum)
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
