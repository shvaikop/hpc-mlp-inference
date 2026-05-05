#include "MLPModel.hpp"
#include "CLI11.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>

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

struct Config {
    std::string export_dir;
    std::string input_path;
    std::string logits_out = "cpp_output_logits.bin";
    std::string preds_out = "cpp_output_preds.bin";
    std::size_t batch_size = 128;
};

int main(int argc, char **argv)
{
    CLI::App app{"MLP Inference"};
    argv = app.ensure_utf8(argv);

    Config cfg;
    app.add_option("--export-dir", cfg.export_dir,  "Path to model export directory")->required();
    app.add_option("--input",      cfg.input_path,  "Path to input embeddings .bin file")->required();
    app.add_option("--logits-out", cfg.logits_out,  "Path to output logits .bin file");
    app.add_option("--preds-out",  cfg.preds_out,   "Path to output predictions .bin file");
    app.add_option("--batch-size", cfg.batch_size,  "Batch size for inference");
    
    CLI11_PARSE(app, argc, argv);

    try
    {
        MLPModel model;
        model.load_from_export_dir(cfg.export_dir);

        std::vector<float> input = read_float32_file(cfg.input_path);
        if (input.size() % model.input_dim() != 0)
        {
            throw std::runtime_error(
                "Input file size is not divisible by input_dim=" + std::to_string(model.input_dim()));
        }

        input.resize(cfg.batch_size * model.input_dim());

        std::cout << "Loaded model with input_dim=" << model.input_dim()
                  << ", num_classes=" << model.num_classes()
                  << ", batch_size=" << cfg.batch_size << "\n";

        const std::vector<float> logits = model.infer_batch(input, cfg.batch_size);
        const std::vector<std::int64_t> preds = model.predict_batch(logits, cfg.batch_size);

        write_float32_file(cfg.logits_out, logits);
        write_int64_file(cfg.preds_out, preds);

        std::cout << "Wrote logits to: " << cfg.logits_out << "\n";
        std::cout << "Wrote preds  to: " << cfg.preds_out << "\n";

        const auto &class_names = model.class_names();
        std::cout << "\nFirst 10 predictions:\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(25, preds.size()); ++i)
        {
            const auto cls = static_cast<std::size_t>(preds[i]);
            std::cout << "  sample " << i << ": class_id=" << preds[i];
            if (cls < class_names.size())
            {
                std::cout << " (" << class_names[cls] << ")";
            }
            std::cout << "\n";
        }

        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
