#ifndef FLAT_MATRIX_VIEW_HPP
#define FLAT_MATRIX_VIEW_HPP

#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "MatrixConcepts.hpp"

template <typename T>
class FlatMatrixView {
public:
    using value_type = T;

    FlatMatrixView() = default;

    FlatMatrixView(std::span<T> data, std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(data)
    {
        if (data_.size() != rows_ * cols_) {
            throw std::runtime_error("FlatMatrixView size does not match dimensions.");
        }
    }

    FlatMatrixView(T* data, std::size_t rows, std::size_t cols)
        : FlatMatrixView(std::span<T>(data, rows * cols), rows, cols)
    {
    }

    template <typename Alloc>
    requires (!std::is_const_v<T>)
    FlatMatrixView(std::vector<T, Alloc>& vec, std::size_t rows, std::size_t cols)
        : FlatMatrixView(std::span<T>(vec.data(), vec.size()), rows, cols)
    {
    }

    template <typename Alloc>
    requires std::is_const_v<T>
    FlatMatrixView(
        const std::vector<std::remove_const_t<T>, Alloc>& vec,
        std::size_t rows,
        std::size_t cols
    )
        : FlatMatrixView(std::span<T>(vec.data(), vec.size()), rows, cols)
    {
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

    T* data() const noexcept
    {
        return data_.data();
    }

    std::span<T> span() const noexcept
    {
        return data_;
    }

    T& operator()(std::size_t r, std::size_t c) const
    {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix index out of range.");
        }

        return data_[r * cols_ + c];
    }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::span<T> data_;
};

static_assert(IMatrix<FlatMatrixView<float>, float>);
static_assert(IMatrix<FlatMatrixView<const float>, float>);
static_assert(WritableMatrix<FlatMatrixView<float>>);
static_assert(!WritableMatrix<FlatMatrixView<const float>>);

#endif // FLAT_MATRIX_VIEW_HPP
