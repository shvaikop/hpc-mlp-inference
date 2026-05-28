#include "CudaMLPModel.hpp"
#include "CLI11.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

static std::vector<float> read_float32_file(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Failed to open input file: " + path.string());

    const std::streamsize size_bytes = file.tellg();
    if (size_bytes < 0 || size_bytes % static_cast<std::streamsize>(sizeof(float)) != 0)
        throw std::runtime_error("Invalid float32 file: " + path.string());

    file.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<std::size_t>(size_bytes) / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(data.data()), size_bytes))
        throw std::runtime_error("Failed to read input file: " + path.string());
    return data;
}

static void write_float32_file(const fs::path& path, const std::vector<float>& data)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open output file for writing: " + path.string());
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
}

static void write_int64_file(const fs::path& path, const std::vector<std::int64_t>& data)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open output file for writing: " + path.string());
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(std::int64_t)));
}

struct Config
{
    std::string export_dir;
    std::string input_path;
    std::string logits_out = "cuda_output_logits.bin";
    std::string preds_out  = "cuda_output_preds.bin";
    std::size_t batch_size = 128;
    bool quiet = false;
};

int main(int argc, char** argv)
{
    CLI::App app{"MLP Inference (CUDA)"};
    argv = app.ensure_utf8(argv);

    Config cfg;
    app.add_option("--export-dir", cfg.export_dir, "Path to model export directory")->required();
    app.add_option("--input",      cfg.input_path, "Path to input embeddings .bin file")->required();
    app.add_option("--logits-out", cfg.logits_out, "Path to output logits .bin file")->capture_default_str();
    app.add_option("--preds-out",  cfg.preds_out,  "Path to output predictions .bin file")->capture_default_str();
    app.add_option("--batch-size", cfg.batch_size, "Batch size for inference")->capture_default_str();
    app.add_flag("--quiet",        cfg.quiet,      "Suppress per-batch output");

    CLI11_PARSE(app, argc, argv);

    try {

        // Load model — weights are uploaded to GPU here
        CudaMLPModel model;
        model.load_from_export_dir(cfg.export_dir);

        // Load all input embeddings
        const std::vector<float> input = read_float32_file(cfg.input_path);
        if (input.size() % model.input_dim() != 0)
            throw std::runtime_error(
                "Input size not divisible by input_dim=" + std::to_string(model.input_dim()));

        const std::size_t total_samples = input.size() / model.input_dim();
        const std::size_t num_batches   = total_samples / cfg.batch_size;
        const std::size_t stride        = cfg.batch_size * model.input_dim();

        std::cout << "================================================================================\n";
        std::cout << "  Input dim    : " << model.input_dim()   << "\n";
        std::cout << "  Num classes  : " << model.num_classes() << "\n";
        std::cout << "  Batch size   : " << cfg.batch_size      << "\n";
        std::cout << "  Total samples: " << total_samples       << "\n";
        std::cout << "  Total batches: " << num_batches         << "\n";
        std::cout << "================================================================================\n";

        std::vector<float>        all_logits;
        std::vector<std::int64_t> all_preds;
        std::vector<double>       batch_times;

        all_logits.reserve(num_batches * cfg.batch_size * model.num_classes());
        all_preds.reserve(num_batches * cfg.batch_size);
        batch_times.reserve(num_batches);

        std::vector<float> logits_buf(cfg.batch_size * model.num_classes());

        for (std::size_t i = 0; i < num_batches; ++i) {
            const float* batch_start = input.data() + i * stride;

            const auto t_start = std::chrono::high_resolution_clock::now();
            // Run inference for the batch, results go into logits_buf
            model.infer_batch(batch_start, logits_buf.data(), cfg.batch_size);

            const auto t_end = std::chrono::high_resolution_clock::now();

            const double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            batch_times.push_back(ms);

            // argmax → predictions
            for (std::size_t b = 0; b < cfg.batch_size; ++b) {
                const float* row = logits_buf.data() + b * model.num_classes();
                const std::int64_t pred = static_cast<std::int64_t>(
                    std::max_element(row, row + model.num_classes()) - row);
                all_preds.push_back(pred);
            }
            all_logits.insert(all_logits.end(), logits_buf.begin(), logits_buf.end());

            if (!cfg.quiet)
                std::cout << "Processed batch " << (i + 1) << "/" << num_batches
                          << " in " << ms << " ms\n";
        }

        // Stats
        const double mean_ms = std::accumulate(batch_times.begin(), batch_times.end(), 0.0)
                               / static_cast<double>(num_batches);

        double variance = 0.0;
        for (const double t : batch_times)
            variance += (t - mean_ms) * (t - mean_ms);
        const double std_ms = std::sqrt(variance / static_cast<double>(num_batches));

        const double throughput = cfg.batch_size / (mean_ms / 1000.0);

        std::cout << "================================================================================\n";
        std::cout << "  Benchmark Results (CUDA)\n";
        std::cout << "================================================================================\n";
        std::cout << "  Mean latency : " << mean_ms   << " ms\n";
        std::cout << "  Std dev      : " << std_ms    << " ms\n";
        std::cout << "  Throughput   : " << throughput << " samples/sec\n";
        std::cout << "================================================================================\n";

        write_float32_file(cfg.logits_out, all_logits);
        write_int64_file(cfg.preds_out, all_preds);
        std::cout << "Wrote logits to: " << cfg.logits_out << "\n";
        std::cout << "Wrote preds  to: " << cfg.preds_out  << "\n";

        if (!cfg.quiet) {
            const auto& class_names = model.class_names();
            std::cout << "\nFirst 25 predictions:\n";
            for (std::size_t i = 0; i < std::min<std::size_t>(25, all_preds.size()); ++i) {
                const auto cls = static_cast<std::size_t>(all_preds[i]);
                std::cout << "  sample " << i << ": class_id=" << all_preds[i];
                if (cls < class_names.size())
                    std::cout << " (" << class_names[cls] << ")";
                std::cout << "\n";
            }
        }

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
