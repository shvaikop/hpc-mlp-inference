#include "MatrixConcepts.hpp"
#include "MLPModel.hpp"
#include "FlatMatrix.hpp"
#include "json.hpp"

#include <algorithm>
#include <stdexcept>

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

std::vector<float> MLPModel::infer_batch(const std::vector<float>& input, std::size_t batch_size) const {
    if (input.size() != batch_size * input_dim_) {
        throw std::runtime_error(
            "Input size mismatch: got " + std::to_string(input.size()) +
            ", expected " + std::to_string(batch_size * input_dim_));
    }

    std::vector<float> activations = input;
    std::size_t current_dim = input_dim_;

    for (std::size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        const bool apply_relu = (layer_idx + 1 != layers_.size());
        activations = linear_forward<FlatMatrix<float>>(activations, batch_size, current_dim, layers_[layer_idx], apply_relu);
        current_dim = layers_[layer_idx].out_features;
    }

    return activations;
}

std::vector<std::int64_t> MLPModel::predict_batch(const std::vector<float>& logits, std::size_t batch_size) const {
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
