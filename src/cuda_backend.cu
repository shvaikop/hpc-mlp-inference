#include "CudaMLPModel.hpp"
#include "json.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// Error checking helpers
// -----------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            throw std::runtime_error(                                           \
                std::string("CUDA error at " __FILE__ ":") +                   \
                std::to_string(__LINE__) + ": " +                               \
                cudaGetErrorString(_err));                                       \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _st = (call);                                            \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                     \
            throw std::runtime_error(                                           \
                std::string("cuBLAS error at " __FILE__ ":") +                 \
                std::to_string(__LINE__) + " status=" + std::to_string(_st));   \
        }                                                                       \
    } while (0)

// -----------------------------------------------------------------------------
// CUDA kernel: add bias row-vector and apply ReLU in-place
//
// output: [batch_size, out_features] row-major (already written by cuBLAS)
// bias:   [out_features]
// -----------------------------------------------------------------------------

__global__ void bias_relu_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int out_features,
    int batch_size
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // feature index
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch index

    if (row < batch_size && col < out_features) {
        const int idx = row * out_features + col;
        float val = output[idx] + bias[col];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

// Same as above but without ReLU (for the last layer)
__global__ void bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int out_features,
    int batch_size
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < out_features) {
        output[row * out_features + col] += bias[col];
    }
}

// -----------------------------------------------------------------------------
// CudaMLPModel — constructor / destructor
// -----------------------------------------------------------------------------

CudaMLPModel::CudaMLPModel() {
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
}

CudaMLPModel::~CudaMLPModel() {
    for (auto& layer : layers_) {
        if (layer.d_weight) cudaFree(layer.d_weight);
        if (layer.d_bias)   cudaFree(layer.d_bias);
    }
    if (cublas_handle_) cublasDestroy(cublas_handle_);
}

// -----------------------------------------------------------------------------
// CudaMLPModel::load_from_export_dir
// Reads metadata.json, then uploads each layer's weight/bias to GPU.
// -----------------------------------------------------------------------------

void CudaMLPModel::load_from_export_dir(const fs::path& export_dir) {
    // Parse metadata
    const fs::path meta_path = export_dir / "metadata.json";
    std::ifstream meta_file(meta_path);
    if (!meta_file)
        throw std::runtime_error("Failed to open metadata file: " + meta_path.string());

    nlohmann::json meta;
    meta_file >> meta;

    if (meta.at("format") != "mlp_classifier_v1")
        throw std::runtime_error("Unsupported model format.");
    if (meta.at("dtype") != "float32")
        throw std::runtime_error("Only float32 dtype supported.");

    const auto& layer_specs = meta.at("layers");

    for (const auto& spec : layer_specs) {
        const std::size_t in_f  = spec.at("in_features");
        const std::size_t out_f = spec.at("out_features");

        // Load from disk
        const std::vector<float> weight = read_binary_file<float>(
            export_dir / std::string(spec.at("weight_file")));
        const std::vector<float> bias = read_binary_file<float>(
            export_dir / std::string(spec.at("bias_file")));

        if (weight.size() != out_f * in_f)
            throw std::runtime_error("Weight size mismatch.");
        if (bias.size() != out_f)
            throw std::runtime_error("Bias size mismatch.");

        // Allocate and upload to GPU
        CudaLinearLayer gpu_layer;
        gpu_layer.in_features  = in_f;
        gpu_layer.out_features = out_f;

        CUDA_CHECK(cudaMalloc(&gpu_layer.d_weight, weight.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(gpu_layer.d_weight, weight.data(),
                              weight.size() * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&gpu_layer.d_bias, bias.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(gpu_layer.d_bias, bias.data(),
                              bias.size() * sizeof(float), cudaMemcpyHostToDevice));

        layers_.push_back(gpu_layer);
    }

    // Derive dimensions from layers
    if (layers_.empty())
        throw std::runtime_error("No layers loaded.");
    input_dim_   = layers_.front().in_features;
    num_classes_ = layers_.back().out_features;

    // Parse class names
    if (meta.contains("class_names")) {
        for (const auto& name : meta.at("class_names"))
            class_names_.push_back(name);
    }
}

// -----------------------------------------------------------------------------
// CudaMLPModel::infer_batch
//
// X[batch_size, in]  →  layer 0  →  ReLU  →  ...  →  layer N-1  →  Y[batch_size, out]
//
// cuBLAS GEMM convention (all row-major via the transpose trick):
//   Y = X * W^T
//   In column-major cuBLAS: op(W_cm)^T * X_cm  →  Y_cm
//
//   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
//               N, M, K,
//               &alpha, d_W, K,   // W [N,K] row-major → col-major [K,N], lda=K
//                       d_X, K,   // X [M,K] row-major → col-major [K,M], lda=K
//               &beta,  d_Y, N);  // Y [M,N] row-major → col-major [N,M], lda=N
// -----------------------------------------------------------------------------

void CudaMLPModel::infer_batch(
    const float* h_input,
    float* h_output,
    std::size_t batch_size
) const {
    const int M = static_cast<int>(batch_size);

    // Allocate two ping-pong buffers on GPU so we avoid allocating every layer
    std::size_t max_features = 0;
    for (const auto& l : layers_)
        max_features = std::max(max_features, std::max(l.in_features, l.out_features));

    float* d_buf[2] = {nullptr, nullptr};
    const std::size_t buf_size = batch_size * max_features * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_buf[0], buf_size));
    CUDA_CHECK(cudaMalloc(&d_buf[1], buf_size));

    // Upload input into buf[0]
    CUDA_CHECK(cudaMemcpy(d_buf[0], h_input,
                          batch_size * input_dim_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;
    const bool last_layer_idx = layers_.size() - 1;

    for (std::size_t li = 0; li < layers_.size(); ++li) {
        const auto& layer = layers_[li];
        const int K = static_cast<int>(layer.in_features);
        const int N = static_cast<int>(layer.out_features);

        float* d_in  = d_buf[li % 2];
        float* d_out = d_buf[(li + 1) % 2];

        // Y = X * W^T  via cuBLAS
        CUBLAS_CHECK(cublasSgemm(
            cublas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            layer.d_weight, K,
            d_in,           K,
            &beta,
            d_out,          N
        ));

        // Add bias + ReLU (skip ReLU on last layer)
        constexpr int TILE = 16;
        const dim3 block(TILE, TILE);
        const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

        const bool is_last = (li == last_layer_idx);
        if (is_last) {
            bias_kernel<<<grid, block>>>(d_out, layer.d_bias, N, M);
        } else {
            bias_relu_kernel<<<grid, block>>>(d_out, layer.d_bias, N, M);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Copy result back to CPU
    const int output_buf = layers_.size() % 2;
    CUDA_CHECK(cudaMemcpy(h_output, d_buf[output_buf],
                          batch_size * num_classes_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_buf[0]);
    cudaFree(d_buf[1]);
}
