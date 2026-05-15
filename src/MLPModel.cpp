#include "MatrixConcepts.hpp"
#include "MLPModel.hpp"
#include "FlatMatrix.hpp"
#include "FlatMatrixView.hpp"
#include "LoggingAllocator.hpp"
#include "FreeListAllocator.hpp"
#include "json.hpp"

#include <algorithm>
#include <stdexcept>
#include <format>

#include "FreeListMemoryPool.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

void MLPModel::load_from_export_dir(const fs::path& export_dir) {
    const fs::path metadata_path = export_dir / "metadata.json";
    std::ifstream meta_file(metadata_path);
    if (!meta_file) {
        throw std::runtime_error("Failed to open metadata file: " + metadata_path.string());
    }

    json meta;
    meta_file >> meta;

    // Validate required metadata fields
    const std::string format = meta.at("format").get<std::string>();
    const std::string dtype = meta.at("dtype").get<std::string>();
    if (format != "mlp_classifier_v1") {
        throw std::runtime_error("Unsupported format: " + format);
    }
    if (dtype != "float32") {
        throw std::runtime_error("Unsupported dtype: " + dtype);
    }

    input_dim_ = meta.at("input_dim").get<std::size_t>();
    num_classes_ = meta.at("num_classes").get<std::size_t>();
    class_names_ = meta.at("class_names").get<std::vector<std::string>>();

    // Loop through the layers and load the wheights
    const auto& layers_json = meta.at("layers");
    layers_.clear();
    layers_.reserve(layers_json.size());

    for (const auto& layer_json : layers_json) {
        const std::string type = layer_json.at("type").get<std::string>();
        if (type != "linear") {
            throw std::runtime_error("Unsupported layer type: " + type);
        }

        LinearLayer layer;
        layer.in_features = layer_json.at("in_features").get<std::size_t>();
        layer.out_features = layer_json.at("out_features").get<std::size_t>();

        const fs::path weight_path = export_dir / layer_json.at("weight_file").get<std::string>();
        const fs::path bias_path = export_dir / layer_json.at("bias_file").get<std::string>();

        layer.weight = read_binary_file<float>(weight_path);
        layer.bias = read_binary_file<float>(bias_path);

        const std::size_t expected_weight_count = layer.out_features * layer.in_features;
        const std::size_t expected_bias_count = layer.out_features;

        if (layer.weight.size() != expected_weight_count) {
            throw std::runtime_error(
                "Weight size mismatch in " + weight_path.string() +
                ": got " + std::to_string(layer.weight.size()) +
                ", expected " + std::to_string(expected_weight_count));
        }
        if (layer.bias.size() != expected_bias_count) {
            throw std::runtime_error(
                "Bias size mismatch in " + bias_path.string() +
                ": got " + std::to_string(layer.bias.size()) +
                ", expected " + std::to_string(expected_bias_count));
        }

        layers_.push_back(std::move(layer));
    }

    if (layers_.empty()) {
        throw std::runtime_error("No layers found in metadata.");
    }
    if (layers_.front().in_features != input_dim_) {
        throw std::runtime_error("First layer input dimension does not match metadata input_dim.");
    }
    if (layers_.back().out_features != num_classes_) {
        throw std::runtime_error("Last layer output dimension does not match metadata num_classes.");
    }
}

std::vector<float, FreeListAllocator<float>> MLPModel::infer_batch(
    const std::vector<float,
    FreeListAllocator<float>>& input,
    std::size_t batch_size,
    FreeListMemoryPool& mem_pool,
    ProfileData& profile_data) const
{
    if (input.size() != batch_size * input_dim_) {
        throw std::runtime_error(
            "Input size mismatch: got " + std::to_string(input.size()) +
            ", expected " + std::to_string(batch_size * input_dim_));
    }
    std::vector<float, FreeListAllocator<float>> activations{FreeListAllocator<float>{mem_pool}};
    std::size_t current_dim = input_dim_;

    for (std::size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        profile_data.append_timestamp(std::format("Layer_{}_start", layer_idx));
        const bool apply_relu = (layer_idx + 1 != layers_.size());
        auto next_activations = linear_forward<FlatMatrix<float, FreeListAllocator<float>>, FlatMatrixView<const float>>(
            layer_idx == 0 ? input : activations,
            batch_size,
            current_dim,
            layers_[layer_idx],
            apply_relu,
            mem_pool,
            profile_data
        );

        profile_data.append_timestamp(std::format("Layer_{}_after_return", layer_idx));

        activations = std::move(next_activations);

        profile_data.append_timestamp(std::format("Layer_{}_after_move_assign", layer_idx));

        current_dim = layers_[layer_idx].out_features;

        profile_data.append_timestamp(std::format("Layer_{}_end", layer_idx));
    }

    return activations;
}

std::vector<std::int64_t> MLPModel::predict_batch(const std::vector<float, FreeListAllocator<float>>& logits, std::size_t batch_size) const {
    if (logits.size() != batch_size * num_classes_) {
        throw std::runtime_error("Logit size mismatch in predict_batch.");
    }

    std::vector<std::int64_t> preds(batch_size, 0);
    for (std::size_t b = 0; b < batch_size; ++b) {
        const float* row = logits.data() + b * num_classes_;
        auto max_it = std::max_element(row, row + num_classes_);
        preds[b] = static_cast<std::int64_t>(std::distance(row, max_it));
    }
    return preds;
}


std::size_t MLPModel::estimate_inference_mem_pool_size(
    std::size_t batch_size
) const {
    if (layers_.empty()) {
        return 0;
    }

    auto activation_payload_bytes = [](std::size_t batch, std::size_t features) {
        if (features != 0 &&
            batch > std::numeric_limits<std::size_t>::max() / features) {
            throw std::overflow_error("Activation element count overflow.");
            }

        const std::size_t elements = batch * features;

        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::overflow_error("Activation byte size overflow.");
        }

        return elements * sizeof(float);
    };

    std::size_t peak_bytes = 0;
    std::size_t previous_features = input_dim_;

    for (const LinearLayer& layer : layers_) {
        const std::size_t previous_bytes =
            activation_payload_bytes(batch_size, previous_features);

        const std::size_t output_bytes =
            activation_payload_bytes(batch_size, layer.out_features);

        if (previous_bytes > std::numeric_limits<std::size_t>::max() - output_bytes) {
            throw std::overflow_error("Live activation byte size overflow.");
        }

        const std::size_t live_bytes = previous_bytes + output_bytes;

        peak_bytes = std::max(peak_bytes, live_bytes);

        previous_features = layer.out_features;
    }

    return peak_bytes;
}
