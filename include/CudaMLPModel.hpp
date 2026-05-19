#ifndef CUDA_MLP_MODEL_HPP
#define CUDA_MLP_MODEL_HPP

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct CudaLinearLayer {
    float* d_weight = nullptr;  // GPU: [out_features, in_features] row-major
    float* d_bias   = nullptr;  // GPU: [out_features]
    std::size_t in_features  = 0;
    std::size_t out_features = 0;
};

class CudaMLPModel {
public:
    CudaMLPModel();
    ~CudaMLPModel();

    CudaMLPModel(const CudaMLPModel&) = delete;
    CudaMLPModel& operator=(const CudaMLPModel&) = delete;

    void load_from_export_dir(const std::filesystem::path& export_dir);

    // Input:  h_input  [batch_size * input_dim]  row-major, on CPU
    // Output: h_output [batch_size * num_classes] row-major, on CPU
    void infer_batch(const float* h_input, float* h_output, std::size_t batch_size) const;

    std::size_t input_dim()   const { return input_dim_; }
    std::size_t num_classes() const { return num_classes_; }
    const std::vector<std::string>& class_names() const { return class_names_; }

private:
    std::size_t input_dim_   = 0;
    std::size_t num_classes_ = 0;
    std::vector<std::string> class_names_;
    std::vector<CudaLinearLayer> layers_;
    cublasHandle_t cublas_handle_ = nullptr;

    template <typename T>
    static std::vector<T> read_binary_file(const std::filesystem::path& path);
};

// -----------------------------------------------------------------------------
// Template implementation
// -----------------------------------------------------------------------------

template <typename T>
std::vector<T> CudaMLPModel::read_binary_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Failed to open binary file: " + path.string());

    const std::streamsize size_bytes = file.tellg();
    if (size_bytes < 0 || size_bytes % static_cast<std::streamsize>(sizeof(T)) != 0)
        throw std::runtime_error("Invalid binary file: " + path.string());

    file.seekg(0, std::ios::beg);
    std::vector<T> data(static_cast<std::size_t>(size_bytes) / sizeof(T));
    if (!file.read(reinterpret_cast<char*>(data.data()), size_bytes))
        throw std::runtime_error("Failed to read binary file: " + path.string());
    return data;
}

#endif // CUDA_MLP_MODEL_HPP
