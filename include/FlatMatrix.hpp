#ifndef FLAT_MATRIX_HPP
#define FLAT_MATRIX_HPP

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "MatrixConcepts.hpp"
#include "FlatMatrixView.hpp"

template <typename T, typename Allocator = std::allocator<T>>
class FlatMatrix {
public:
    using value_type = T;

    FlatMatrix() = default;

    FlatMatrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols)
    {
    }

    // Copies if called with an lvalue vector.
    // Moves if called with std::move(vector).
    FlatMatrix(std::size_t rows, std::size_t cols, std::vector<T, Allocator> data)
        : rows_(rows), cols_(cols), data_(std::move(data))
    {
        if (data_.size() != rows * cols) {
            throw std::runtime_error("Data size does not match matrix dimensions.");
        }
    }

    std::size_t rows() const noexcept
    {
        return rows_;
    }

    std::size_t cols() const noexcept
    {
        return cols_;
    }

    std::size_t size() const noexcept
    {
        return data_.size();
    }

    T* data() noexcept
    {
        return data_.data();
    }

    const T* data() const noexcept
    {
        return data_.data();
    }

    std::vector<T, Allocator>& vector() noexcept
    {
        return data_;
    }

    const std::vector<T, Allocator>& vector() const noexcept
    {
        return data_;
    }

    T& operator()(std::size_t r, std::size_t c)
    {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix index out of range.");
        }

        return data_[r * cols_ + c];
    }

    const T& operator()(std::size_t r, std::size_t c) const
    {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix index out of range.");
        }

        return data_[r * cols_ + c];
    }

    FlatMatrixView<T> view() noexcept
    {
        return FlatMatrixView<T>(data_.data(), rows_, cols_);
    }

    FlatMatrixView<const T> view() const noexcept
    {
        return FlatMatrixView<const T>(data_.data(), rows_, cols_);
    }

    std::vector<T, Allocator> take_data()
    {
        std::vector<T, Allocator> stolen_data = std::move(data_);

        rows_ = 0;
        cols_ = 0;

        return stolen_data;
    }

    void print_statistics() const
    {
        if (data_.empty()) {
            std::cout << "Matrix is empty.\n";
            return;
        }

        const std::size_t total_elements = data_.size();

        const std::size_t non_zero_count =
            std::count_if(data_.begin(), data_.end(), [](T v) {
                return v != T{0};
            });

        const double density =
            static_cast<double>(non_zero_count) / static_cast<double>(total_elements);

        const double sparsity = 1.0 - density;

        auto [min_it, max_it] = std::minmax_element(data_.begin(), data_.end());

        const T min_val = *min_it;
        const T max_val = *max_it;

        const double sum = std::accumulate(data_.begin(), data_.end(), 0.0);
        const double mean = sum / static_cast<double>(total_elements);

        const double sq_sum =
            std::inner_product(data_.begin(), data_.end(), data_.begin(), 0.0);

        const double variance = sq_sum / static_cast<double>(total_elements) - mean * mean;
        const double stdev = std::sqrt(std::max(0.0, variance));

        std::cout << "--- Matrix Statistics ---\n";
        std::cout << "Dimensions: " << rows_ << " x " << cols_ << '\n';
        std::cout << "Density: " << density * 100.0 << "%\n";
        std::cout << "Sparsity: " << sparsity * 100.0 << "%\n";

        std::cout << "\n--- Value Distribution ---\n";
        std::cout << "Min: " << min_val << '\n';
        std::cout << "Max: " << max_val << '\n';
        std::cout << "Mean: " << mean << '\n';
        std::cout << "StdDev: " << stdev << '\n';

        std::cout << "\n--- Range Spread ---\n";

        if (min_val != max_val) {
            constexpr int BUCKETS = 5;

            std::vector<std::size_t> histogram(BUCKETS, 0);

            const double range = static_cast<double>(max_val - min_val);

            for (T val : data_) {
                int bucket =
                    std::min(
                        static_cast<int>(
                            (static_cast<double>(val - min_val) / range) * BUCKETS
                        ),
                        BUCKETS - 1
                    );

                histogram[bucket]++;
            }

            for (int i = 0; i < BUCKETS; ++i) {
                const double lower =
                    static_cast<double>(min_val) + (range / BUCKETS) * i;

                const double upper =
                    static_cast<double>(min_val) + (range / BUCKETS) * (i + 1);

                std::string bar(histogram[i] * 20 / total_elements, '#');

                std::cout << "[" << lower << ", " << upper << "]: "
                  << histogram[i] << " "
                  << bar << '\n';
            }
        } else {
            std::cout << "All values are identical.\n";
        }

        std::cout << "-------------------------\n";
    }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<T, Allocator> data_;
};

static_assert(IMatrix<FlatMatrix<float>, float>);
static_assert(WritableMatrix<FlatMatrix<float>>);

#endif // FLAT_MATRIX_HPP
