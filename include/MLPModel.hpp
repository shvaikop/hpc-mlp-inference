#ifndef MLP_MODEL_HPP
#define MLP_MODEL_HPP

#pragma once

#include "MatrixConcepts.hpp"
#include "utils.hpp"
#include "MatrixAlgorithms.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <span>

struct LinearLayer {
    std::size_t in_features = 0;
    std::size_t out_features = 0;
    std::vector<float> weight; // shape: [out_features, in_features], row-major
    std::vector<float> bias;   // shape: [out_features]
};

class MLPModel {
public:
    void load_from_export_dir(const std::filesystem::path& export_dir);

    std::size_t input_dim() const { return input_dim_; }
    std::size_t num_classes() const { return num_classes_; }
    const std::vector<std::string>& class_names() const { return class_names_; }

    // Input shape:  [batch_size, input_dim]
    // Output shape: [batch_size, num_classes]
    std::vector<float> infer_batch(const std::vector<float>& input, std::size_t batch_size, ProfileData& profile_data) const;

    std::vector<std::int64_t> predict_batch(const std::vector<float>& logits, std::size_t batch_size) const;

private:
    std::size_t input_dim_ = 0;
    std::size_t num_classes_ = 0;
    std::vector<std::string> class_names_;
    std::vector<LinearLayer> layers_;

    template <typename T>
    static std::vector<T> read_binary_file(const std::filesystem::path& path);

    template <typename MatrixType, typename MatrixViewType>
        requires IMatrix<MatrixType, float> && IMatrix<MatrixViewType, float> && WritableMatrix<MatrixType>
    static std::vector<float> linear_forward(
        const std::vector<float>& input,
        std::size_t batch_size,
        std::size_t input_dim,
        const LinearLayer& layer,
        bool apply_relu,
        ProfileData& profile_data
    );
};

// -----------------------------------------------------------------------------
// Template Implementations
// -----------------------------------------------------------------------------

template <typename T>
std::vector<T> MLPModel::read_binary_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open binary file: " + path.string());
    }

    const std::streamsize size_bytes = file.tellg();
    if (size_bytes < 0) {
        throw std::runtime_error("Failed to determine file size: " + path.string());
    }
    if (size_bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("Binary file size is not divisible by element size: " + path.string());
    }

    file.seekg(0, std::ios::beg);
    std::vector<T> data(static_cast<std::size_t>(size_bytes) / sizeof(T));
    if (!file.read(reinterpret_cast<char*>(data.data()), size_bytes)) {
        throw std::runtime_error("Failed to read binary file: " + path.string());
    }
    return data;
}

template <typename MatrixType, typename MatrixViewType>
    requires IMatrix<MatrixType, float> && IMatrix<MatrixViewType, float> && WritableMatrix<MatrixType>
std::vector<float> MLPModel::linear_forward(
    const std::vector<float>& input,
    std::size_t batch_size,
    std::size_t input_dim,
    const LinearLayer& layer,
    bool apply_relu_flag,
    ProfileData& profile_data
) {
    if (input_dim != layer.in_features) {
        throw std::runtime_error("Layer input dimension mismatch.");
    }

    if (input.size() != batch_size * input_dim) {
        throw std::runtime_error("Input size does not match batch_size * input_dim.");
    }

    if (layer.weight.size() != layer.out_features * layer.in_features) {
        throw std::runtime_error("Layer weight size does not match layer dimensions.");
    }

    if (layer.bias.size() != layer.out_features) {
        throw std::runtime_error("Layer bias size does not match output dimension.");
    }

    std::vector<float> y_buffer(batch_size * layer.out_features);

    profile_data.append_timestamp("Before_matrix_init");

    std::span<const float> input_span(
        input.data(),
        input.size()
    );

    std::span<const float> weight_span(
        layer.weight.data(),
        layer.weight.size()
    );

    MatrixViewType X(
        input_span,
        batch_size,
        input_dim
    );

    MatrixViewType W(
        weight_span,
        layer.out_features,
        layer.in_features
    );

    profile_data.append_timestamp("Middle_matrix_init");

    // Owning output matrix: this is the only matrix allocation here.
    MatrixType Y(
        batch_size,
        layer.out_features,
        std::move(y_buffer)
    );

    // std::cout << "X view: "
    //           << batch_size << " x " << input_dim
    //           << ", elements: " << batch_size * input_dim
    //           << ", bytes referenced: " << batch_size * input_dim * sizeof(float)
    //           << ", newly allocated bytes: 0"
    //           << '\n';
    //
    // std::cout << "W view: "
    //           << layer.out_features << " x " << layer.in_features
    //           << ", elements: " << layer.out_features * layer.in_features
    //           << ", bytes referenced: "
    //           << layer.out_features * layer.in_features * sizeof(float)
    //           << ", newly allocated bytes: 0"
    //           << '\n';
    //
    // std::cout << "Y owning: "
    //           << batch_size << " x " << layer.out_features
    //           << ", elements: " << batch_size * layer.out_features
    //           << ", newly allocated bytes: "
    //           << batch_size * layer.out_features * sizeof(float)
    //           << '\n';
    //
    // std::cout << "Total newly allocated bytes: "
    //           << batch_size * layer.out_features * sizeof(float)
    //           << '\n';

    profile_data.append_timestamp("After_matrix_init");

    multiply_transposed_rhs(X, W, Y);
    profile_data.append_timestamp("After_matrix_mul");

    add_row_vector(Y, layer.bias);
    profile_data.append_timestamp("After_matrix_add");

    if (apply_relu_flag) {
        apply_relu(Y);
    }

    profile_data.append_timestamp("After_matrix_relu");

    return Y.take_data();
}

#endif // MLP_MODEL_HPP