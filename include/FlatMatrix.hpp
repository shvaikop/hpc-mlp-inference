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

    void multiply_transposed_rhs(const FlatMatrix& rhs, FlatMatrix& out) const {
        // std::cout << "Multiplying matrix " << rows_ << " x " << cols_ << " by matrix " << rhs.cols() << " x " << rhs.rows() << "\n";
        if (cols_ != rhs.cols_) {
            throw std::runtime_error("Dimension mismatch.");
        }

        if (out.rows() != rows_ || out.cols() != rhs.rows()) {
            throw std::runtime_error("Output matrix has incorrect dimensions.");
        }

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

};


static_assert(IMatrix<FlatMatrix<float>, float>, 
    "FlatMatrix does not perfectly implement the IMatrix concept!");

#endif // FLAT_MATRIX_HPP
