#ifndef FLAT_MATRIX_HPP
#define FLAT_MATRIX_HPP

#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

class FlatMatrix {
public:
    FlatMatrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

    // TODO: check if unnecessary copy is needed
    FlatMatrix(std::size_t rows, std::size_t cols, const std::vector<float>& data)
        : rows_(rows), cols_(cols), data_(data) {
        if (data.size() != rows * cols) {
            throw std::runtime_error("Data size does not match matrix dimensions.");
        }
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }

    const float& operator()(std::size_t r, std::size_t c) const {
        // TODO: experiment if removing bounds check improves performance
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix index out of range.");
        }
        return data_[r * cols_ + c];
    }

    void multiply_transposed_rhs(const FlatMatrix& rhs, FlatMatrix& out) const {
        if (cols_ != rhs.cols_) {
            throw std::runtime_error("Dimension mismatch.");
        }

        if (out.rows() != rows_ || out.cols() != rhs.rows()) {
            throw std::runtime_error("Output matrix has incorrect dimensions.");
        }

        for (std::size_t i = 0; i < rows_; ++i) {
            const float* x = &data_[i * cols_];
            float* y = &out.data_[i * out.cols_];

            for (std::size_t j = 0; j < rhs.rows_; ++j) {
                const float* w = &rhs.data_[j * rhs.cols_];

                float sum = 0.0f;
                for (std::size_t k = 0; k < cols_; ++k) {
                    sum += x[k] * w[k];
                }

                y[j] = sum;
            }
        }
    }

    void add_row_vector(const std::vector<float>& vec) {
        if (vec.size() != cols_) {
            throw std::runtime_error("Vector size does not match number of columns.");
        }

        for (std::size_t r = 0; r < rows_; ++r) {
            float* row = &data_[r * cols_];
            for (std::size_t c = 0; c < cols_; ++c) {
                row[c] += vec[c];
            }
        }
    }

    template <typename Func>
    void transform(Func func) {
        for (float& val : data_) {
            val = func(val);
        }
    }

    std::vector<float>&& take_data() {
        return std::move(data_);
    }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<float> data_;   // row-major storage

};


#endif // FLAT_MATRIX_HPP