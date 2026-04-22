#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

struct LinearLayer {
    std::size_t in_features = 0;
    std::size_t out_features = 0;
    std::vector<float> weight; // shape: [out_features, in_features], row-major
    std::vector<float> bias;   // shape: [out_features]
};

class MLPModel {
public:
    void load_from_export_dir(const fs::path& export_dir) {
        const fs::path metadata_path = export_dir / "metadata.json";
        std::ifstream meta_file(metadata_path);
        if (!meta_file) {
            throw std::runtime_error("Failed to open metadata file: " + metadata_path.string());
        }

        json meta;
        meta_file >> meta;

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

    std::size_t input_dim() const { return input_dim_; }
    std::size_t num_classes() const { return num_classes_; }
    const std::vector<std::string>& class_names() const { return class_names_; }

    // Input shape:  [batch_size, input_dim]
    // Output shape: [batch_size, num_classes]
    std::vector<float> infer_batch(const std::vector<float>& input, std::size_t batch_size) const {
        if (input.size() != batch_size * input_dim_) {
            throw std::runtime_error(
                "Input size mismatch: got " + std::to_string(input.size()) +
                ", expected " + std::to_string(batch_size * input_dim_));
        }

        std::vector<float> activations = input;
        std::size_t current_dim = input_dim_;

        for (std::size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
            const bool apply_relu = (layer_idx + 1 != layers_.size());
            activations = linear_forward(activations, batch_size, current_dim, layers_[layer_idx], apply_relu);
            current_dim = layers_[layer_idx].out_features;
        }

        return activations;
    }

    std::vector<std::int64_t> predict_batch(const std::vector<float>& logits, std::size_t batch_size) const {
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

private:
    std::size_t input_dim_ = 0;
    std::size_t num_classes_ = 0;
    std::vector<std::string> class_names_;
    std::vector<LinearLayer> layers_;

    template <typename T>
    static std::vector<T> read_binary_file(const fs::path& path) {
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

    static std::vector<float> linear_forward(
        const std::vector<float>& input,
        std::size_t batch_size,
        std::size_t input_dim,
        const LinearLayer& layer,
        bool apply_relu
    ) {
        if (input_dim != layer.in_features) {
            throw std::runtime_error("Layer input dimension mismatch.");
        }

        std::vector<float> output(batch_size * layer.out_features, 0.0f);

        for (std::size_t b = 0; b < batch_size; ++b) {
            const float* x = input.data() + b * input_dim;
            float* y = output.data() + b * layer.out_features;

            for (std::size_t o = 0; o < layer.out_features; ++o) {
                const float* w_row = layer.weight.data() + o * layer.in_features;
                float sum = layer.bias[o];

                for (std::size_t i = 0; i < layer.in_features; ++i) {
                    sum += x[i] * w_row[i];
                }

                if (apply_relu && sum < 0.0f) {
                    sum = 0.0f;
                }
                y[o] = sum;
            }
        }

        return output;
    }
};

static std::vector<float> read_float32_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open input file: " + path.string());
    }

    const std::streamsize size_bytes = file.tellg();
    if (size_bytes < 0 || size_bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("Invalid float32 file: " + path.string());
    }

    file.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<std::size_t>(size_bytes) / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(data.data()), size_bytes)) {
        throw std::runtime_error("Failed to read input file: " + path.string());
    }
    return data;
}

static void write_float32_file(const fs::path& path, const std::vector<float>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open output file for writing: " + path.string());
    }
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
}

static void write_int64_file(const fs::path& path, const std::vector<std::int64_t>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open output file for writing: " + path.string());
    }
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(std::int64_t)));
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr
                << "Usage:\n"
                << "  " << argv[0] << " <export_dir> <input_embeddings.bin> [output_logits.bin] [output_preds.bin]\n\n"
                << "Example:\n"
                << "  " << argv[0] << " export test_input_embeddings.bin logits.bin preds.bin\n";
            return 1;
        }

        const fs::path export_dir = argv[1];
        const fs::path input_path = argv[2];
        const fs::path logits_out_path = (argc >= 4) ? fs::path(argv[3]) : fs::path("cpp_output_logits.bin");
        const fs::path preds_out_path  = (argc >= 5) ? fs::path(argv[4]) : fs::path("cpp_output_preds.bin");

        MLPModel model;
        model.load_from_export_dir(export_dir);

        std::vector<float> input = read_float32_file(input_path);
        if (input.size() % model.input_dim() != 0) {
            throw std::runtime_error(
                "Input file size is not divisible by input_dim=" + std::to_string(model.input_dim()));
        }

        // TODO: Change to be a parameter
        const std::size_t batch_size = 128;
        input.resize(batch_size * model.input_dim());

        // const std::size_t batch_size = input.size() / model.input_dim();
        std::cout << "Loaded model with input_dim=" << model.input_dim()
                  << ", num_classes=" << model.num_classes()
                  << ", batch_size=" << batch_size << "\n";

        const std::vector<float> logits = model.infer_batch(input, batch_size);
        const std::vector<std::int64_t> preds = model.predict_batch(logits, batch_size);

        write_float32_file(logits_out_path, logits);
        write_int64_file(preds_out_path, preds);

        std::cout << "Wrote logits to: " << logits_out_path << "\n";
        std::cout << "Wrote preds  to: " << preds_out_path << "\n";

        const auto& class_names = model.class_names();
        std::cout << "\nFirst 10 predictions:\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(25, preds.size()); ++i) {
            const auto cls = static_cast<std::size_t>(preds[i]);
            std::cout << "  sample " << i << ": class_id=" << preds[i];
            if (cls < class_names.size()) {
                std::cout << " (" << class_names[cls] << ")";
            }
            std::cout << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
