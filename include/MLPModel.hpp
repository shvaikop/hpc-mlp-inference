#ifndef MLP_MODEL_HPP
#define MLP_MODEL_HPP

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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
    std::vector<float> infer_batch(const std::vector<float>& input, std::size_t batch_size) const;

    std::vector<std::int64_t> predict_batch(const std::vector<float>& logits, std::size_t batch_size) const;

private:
    std::size_t input_dim_ = 0;
    std::size_t num_classes_ = 0;
    std::vector<std::string> class_names_;
    std::vector<LinearLayer> layers_;

    template <typename T>
    static std::vector<T> read_binary_file(const std::filesystem::path& path);

    static std::vector<float> linear_forward(
        const std::vector<float>& input,
        std::size_t batch_size,
        std::size_t input_dim,
        const LinearLayer& layer,
        bool apply_relu
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

#endif // MLP_MODEL_HPP