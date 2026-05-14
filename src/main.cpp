#include "MLPModel.hpp"
#include "CLI11.hpp"

#include <optional>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <numeric>
#include <cmath>

namespace fs = std::filesystem;

static std::vector<float> read_float32_file(const fs::path &path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Failed to open input file: " + path.string());
    }

    const std::streamsize size_bytes = file.tellg();
    if (size_bytes < 0 || size_bytes % static_cast<std::streamsize>(sizeof(float)) != 0)
    {
        throw std::runtime_error("Invalid float32 file: " + path.string());
    }

    file.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<std::size_t>(size_bytes) / sizeof(float));
    if (!file.read(reinterpret_cast<char *>(data.data()), size_bytes))
    {
        throw std::runtime_error("Failed to read input file: " + path.string());
    }
    return data;
}

static void write_float32_file(const fs::path &path, const std::vector<float> &data)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open output file for writing: " + path.string());
    }
    file.write(reinterpret_cast<const char *>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
}

static void write_int64_file(const fs::path &path, const std::vector<std::int64_t> &data)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open output file for writing: " + path.string());
    }
    file.write(reinterpret_cast<const char *>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(std::int64_t)));
}

struct Config
{
    static constexpr std::string_view DEFAULT_LOGITS_OUT = "cpp_output_logits.bin";
    static constexpr std::string_view DEFAULT_PREDS_OUT = "cpp_output_preds.bin";
    static constexpr std::size_t DEFAULT_BATCH_SIZE = 128;
    static constexpr std::optional<std::size_t> MAX_NUM_BATCHES_PLACEHOLDER = std::nullopt;

    std::string export_dir;
    std::string input_path;
    std::string logits_out{DEFAULT_LOGITS_OUT};
    std::string preds_out{DEFAULT_PREDS_OUT};
    std::size_t batch_size = DEFAULT_BATCH_SIZE;
    std::optional<std::size_t> num_batches = MAX_NUM_BATCHES_PLACEHOLDER;
    bool quiet = false;
};

int main(int argc, char **argv)
{
    CLI::App app{"MLP Inference"};
    argv = app.ensure_utf8(argv);

    Config cfg;
    app.add_option("--export-dir", cfg.export_dir, "Path to model export directory")->required();
    app.add_option("--input", cfg.input_path, "Path to input embeddings .bin file")->required();
    app.add_option("--logits-out", cfg.logits_out, "Path to output logits .bin file")->capture_default_str();
    app.add_option("--preds-out", cfg.preds_out, "Path to output predictions .bin file")->capture_default_str();
    app.add_option("--batch-size", cfg.batch_size, "Batch size for inference")->capture_default_str();
    app.add_option("--num-batches", cfg.num_batches, "Number of batches to run")->default_str("All samples from dataset");
    app.add_flag("--quiet", cfg.quiet, "Suppress per-batch output, show only final summary");

    CLI11_PARSE(app, argc, argv);

    try
    {
        // Load the model
        MLPModel model;
        model.load_from_export_dir(cfg.export_dir);

        // Load the input embeddings
        std::vector<float> input = read_float32_file(cfg.input_path);
        if (input.size() % model.input_dim() != 0)
        {
            throw std::runtime_error(
                "Input file size is not divisible by input_dim=" + std::to_string(model.input_dim()));
        }

        // Calculate the number of samples and batch
        // Lambda is used so that `num_samples` can be marked const and its more readable than with ternary operator
        const std::size_t num_samples = [&]()
        {
            if (!cfg.num_batches.has_value())
            {
                const std::size_t total_samples = input.size() / model.input_dim();
                cfg.num_batches = total_samples / cfg.batch_size;
                return total_samples;
            }
            return cfg.num_batches.value() * cfg.batch_size;
        }();

        std::cout << "================================================================================\n";
        std::cout << "  Input dim    : " << model.input_dim() << "\n";
        std::cout << "  Num classes  : " << model.num_classes() << "\n";
        std::cout << "  Batch size   : " << cfg.batch_size << "\n";
        std::cout << "  Total samples: " << num_samples << "\n";
        std::cout << "  Total batches: " << cfg.num_batches.value() << "\n";
        std::cout << "================================================================================\n";

        std::vector<double> batch_times;
        const std::size_t stride = cfg.batch_size * model.input_dim();

        std::vector<float> all_logits;
        std::vector<std::int64_t> all_preds;

        // Iterate over batches and run inference
        for (std::size_t i = 0; i < cfg.num_batches.value(); ++i)
        {
            // Create batch input
            const float *batch_start = input.data() + i * stride;
            const std::vector<float> batch_input(batch_start, batch_start + stride);

            // Calculate the time for each batch inference
            const auto t_start = std::chrono::high_resolution_clock::now();
            const std::vector<float> logits = model.infer_batch(batch_input, cfg.batch_size);
            const auto t_end = std::chrono::high_resolution_clock::now();

            batch_times.push_back(std::chrono::duration<double, std::milli>(t_end - t_start).count());

            const std::vector<std::int64_t> preds = model.predict_batch(logits, cfg.batch_size);
            all_logits.insert(all_logits.end(), logits.begin(), logits.end());
            all_preds.insert(all_preds.end(), preds.begin(), preds.end());
            if (!cfg.quiet)
                std::cout << "Processed batch " << (i + 1) << "/" << cfg.num_batches.value() << " in " << batch_times.back() << " ms\n";
        }

        // average latency
        const double mean_ms = std::accumulate(batch_times.begin(), batch_times.end(), 0.0) / cfg.num_batches.value();

        // standard deviation
        double variance = 0.0;
        for (const double t : batch_times)
        {
            variance += (t - mean_ms) * (t - mean_ms);
        }
        const double std_ms = std::sqrt(variance / cfg.num_batches.value());

        const double throughput = cfg.batch_size / (mean_ms / 1000.0);

        std::cout << "================================================================================\n";
        std::cout << "  Benchmark Results\n";
        std::cout << "================================================================================\n";
        std::cout << "  Mean latency : " << mean_ms << " ms\n";
        std::cout << "  Std dev      : " << std_ms << " ms\n";
        std::cout << "  Throughput   : " << throughput << " samples/sec\n";
        std::cout << "================================================================================\n";

        // write results to files
        write_float32_file(cfg.logits_out, all_logits);
        write_int64_file(cfg.preds_out, all_preds);

        std::cout << "Wrote logits to: " << cfg.logits_out << "\n";
        std::cout << "Wrote preds  to: " << cfg.preds_out << "\n";

        if (!cfg.quiet) {
            const auto &class_names = model.class_names();
            std::cout << "\nFirst 10 predictions:\n";
            for (std::size_t i = 0; i < std::min<std::size_t>(25, all_preds.size()); ++i)
            {
                const auto cls = static_cast<std::size_t>(all_preds[i]);
                std::cout << "  sample " << i << ": class_id=" << all_preds[i];
                if (cls < class_names.size())
                {
                    std::cout << " (" << class_names[cls] << ")";
                }
                std::cout << "\n";
            }
        }

        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
