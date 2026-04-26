#ifndef MATRIX_CONCEPTS_HPP
#define MATRIX_CONCEPTS_HPP

#pragma once

#include <concepts>
#include <vector>
#include <cstddef>

// ---------------------------------------------------------
// Matrix interface concept
// ---------------------------------------------------------
template <typename M, typename T = float>
concept IMatrix = requires(M mutable_m, const M const_m, M out_m, const std::vector<T>& vec) {
    // 1. Dimension accessors must exist and return a size type
    { const_m.rows() } -> std::convertible_to<std::size_t>;
    { const_m.cols() } -> std::convertible_to<std::size_t>;

    // 2. Element access operator must exist
    { const_m(0, 0) } -> std::convertible_to<T>;

    // 3. Core Math Operations
    // Must be able to multiply and store result in an out-parameter
    const_m.multiply_transposed_rhs(const_m, out_m);
    
    // Must be able to broadcast a row vector
    mutable_m.add_row_vector(vec);

    // 4. Functional mapping
    // Must accept a lambda that takes T and returns T
    mutable_m.transform([](T x) { return x; });

    // 5. Data extraction
    // STRICT REQUIREMENT: Must return exactly std::vector<T> by value to ensure safe moves.
    { mutable_m.take_data() } -> std::same_as<std::vector<T>>;
};

#endif // MATRIX_CONCEPTS_HPP
