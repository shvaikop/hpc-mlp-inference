#ifndef MATRIX_CONCEPTS_HPP
#define MATRIX_CONCEPTS_HPP

#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

template <typename M>
concept MatrixLike = requires(M& m, const M& cm) {
    typename std::remove_cvref_t<M>::value_type;

    { cm.rows() } -> std::convertible_to<std::size_t>;
    { cm.cols() } -> std::convertible_to<std::size_t>;
    { cm.size() } -> std::convertible_to<std::size_t>;

    { m.data() };
    { cm.data() };
};

template <typename M>
using matrix_value_t = typename std::remove_cvref_t<M>::value_type;

template <typename M>
using matrix_scalar_t = std::remove_cv_t<matrix_value_t<M>>;

template <typename M, typename T>
concept IMatrix =
    MatrixLike<M> &&
    std::same_as<matrix_scalar_t<M>, T>;

template <typename M>
concept WritableMatrix =
    MatrixLike<M> &&
    std::is_pointer_v<decltype(std::declval<M&>().data())> &&
    !std::is_const_v<
        std::remove_pointer_t<decltype(std::declval<M&>().data())>
    >;

#endif // MATRIX_CONCEPTS_HPP
