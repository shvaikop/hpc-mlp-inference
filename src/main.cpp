#include "MLPModel.hpp"
#include "CLI11.hpp"
#include "utils.hpp"
#include "LoggingAllocator.hpp"
#include "FreeListMemoryPool.hpp"
#include "FreeListAllocator.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <optional>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <numeric>
#include <cmath>

namespace fs = std::filesystem;
// Utility functions for reading/writing binary files of float32 and int64 types
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
// Similar function for float32 files
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

// Similar function for int64 files
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

// Configuration struct to hold command-line arguments
struct Config
{
    static constexpr std::string_view DEFAULT_LOGITS_OUT = "cpp_output_logits.bin";
    static constexpr std::string_view DEFAULT_PREDS_OUT = "cpp_output_preds.bin";
    static constexpr std::size_t DEFAULT_BATCH_SIZE = 128; // Default batch size if not specified
    static constexpr std::optional<std::size_t> MAX_NUM_BATCHES_PLACEHOLDER = std::nullopt;

    std::string export_dir; // Path to model weights and config
    std::string input_path; // Path to input embeddings .bin file
    std::string logits_out{DEFAULT_LOGITS_OUT}; // Path to output logits .bin file
    std::string preds_out{DEFAULT_PREDS_OUT}; // Path to output predictions .bin file
    std::size_t batch_size = DEFAULT_BATCH_SIZE; 
    std::optional<std::size_t> num_batches = MAX_NUM_BATCHES_PLACEHOLDER; // Number of batches to run, if not specified, will run on all samples in the input dataset
    bool quiet = false;
};

int main(int argc, char **argv)
{
#ifdef USE_MPI
    // Initialize MPI 
    MPI_Init(&argc, &argv);
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#else
    constexpr int mpi_rank = 0;
    constexpr int mpi_size = 1;
#endif

    // ============== Command-line argument parsing using CLI11 ==============
    // srun is responsible to pass the command-line arguments to all ranks
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
    // =====================================================================
    try
    {
        ProfileData profile_data{};
        profile_data.append_timestamp("Start");

        // Load the model, every rank does this
        MLPModel model;
        model.load_from_export_dir(cfg.export_dir);
        profile_data.append_timestamp("Model_loaded");

        // Load the input embeddings (only rank 0 reads from disk)
        std::vector<float> input;
        if (mpi_rank == 0) {
            input = read_float32_file(cfg.input_path);
            if (input.size() % model.input_dim() != 0)
            {
                throw std::runtime_error(
                    "Input file size is not divisible by input_dim=" + std::to_string(model.input_dim()));
            }
        }
        profile_data.append_timestamp("Input_loaded");

        // Calculate the number of samples and batch (rank 0 only, then broadcast)
        std::size_t num_samples = 0;
        if (mpi_rank == 0) {
            num_samples = [&]()
            {
                // if not specified, calculate the number of batches based on the input file size and batch size
                if (!cfg.num_batches.has_value())
                {
                    const std::size_t total_samples = input.size() / model.input_dim();
                    cfg.num_batches = total_samples / cfg.batch_size;
                    return total_samples;
                }
                return cfg.num_batches.value() * cfg.batch_size;
            }();
        }
#ifdef USE_MPI
        {
            // Broadcast num_samples to all ranks
            std::size_t nb = cfg.num_batches.value_or(0);
            MPI_Bcast(&nb, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            cfg.num_batches = nb;
        }
#endif
        // Rank 0 prints the configuration summary
        if (mpi_rank == 0) {
            std::cout << "================================================================================\n";
            std::cout << "  Input dim    : " << model.input_dim() << "\n";
            std::cout << "  Num classes  : " << model.num_classes() << "\n";
            std::cout << "  Batch size   : " << cfg.batch_size << "\n";
            std::cout << "  Total samples: " << num_samples << "\n";
            std::cout << "  Total batches: " << cfg.num_batches.value() << "\n";
#ifdef USE_MPI

            std::cout << "  MPI ranks    : " << mpi_size << "\n";
            std::cout << "  Batches/rank : " << cfg.num_batches.value() / mpi_size << "\n";
#endif
            std::cout << "================================================================================\n";
        }

        std::vector<double> batch_times;
        const std::size_t stride = cfg.batch_size * model.input_dim();

        std::vector<float> all_logits;
        std::vector<std::int64_t> all_preds;

#ifdef USE_MPI
        // Scatter input data to all ranks
        const int batches_per_process = static_cast<int>(cfg.num_batches.value()) / mpi_size;

        // stride = batch_size * input_dim , such as 64*512 = 32768 floats per batch
        std::vector<float> local_input(static_cast<std::size_t>(batches_per_process) * stride);
        MPI_Scatter(
            input.data(),
            static_cast<int>(batches_per_process * stride),
            MPI_FLOAT,
            local_input.data(),
            static_cast<int>(batches_per_process * stride),
            MPI_FLOAT,
            0, MPI_COMM_WORLD);
#else
        const int batches_per_process = static_cast<int>(cfg.num_batches.value());
        const std::vector<float>& local_input = input;
#endif

        profile_data.append_timestamp("Before_mem_pool_init");
        FreeListMemoryPool mem_pool{
            model.estimate_inference_mem_pool_size(cfg.batch_size) * 5 / 4
        };
        profile_data.append_timestamp("After_mem_pool_init");

        // Iterate over batches and run inference
        for (std::size_t i = 0; i < static_cast<std::size_t>(batches_per_process); ++i) {
            // Create batch input, zero-copy by pointing to the correct offset in the local_input vector
            const float* batch_start = local_input.data() + i * stride;
            const std::vector<float, FreeListAllocator<float>> batch_input(batch_start, batch_start + stride, FreeListAllocator<float>{mem_pool});

            // Calculate the time for each batch inference
            const auto t_start = std::chrono::system_clock::now();
            profile_data.append_timestamp("Batch_" + std::to_string(i) + "_start", t_start);

            // Run inference for the batch and get logits, then predict classes from logits
            const std::vector<float, FreeListAllocator<float>> logits = model.infer_batch(batch_input, cfg.batch_size, mem_pool, profile_data);
            
            const auto t_end = std::chrono::system_clock::now();
            profile_data.append_timestamp("Batch_" + std::to_string(i) + "_end", t_end);

            batch_times.push_back(std::chrono::duration<double, std::milli>(t_end - t_start).count());

            const std::vector<std::int64_t> preds = model.predict_batch(logits, cfg.batch_size);
            all_logits.insert(all_logits.end(), logits.begin(), logits.end());
            all_preds.insert(all_preds.end(), preds.begin(), preds.end());
            if (!cfg.quiet)
                std::cout << "Processed batch " << (i + 1) << "/" << batches_per_process << " in " << batch_times.back() << " ms\n";
        }

#ifdef USE_MPI
        // Gather batch times, logits, and preds from all ranks to RANK 0!!!
        const int total_batches = static_cast<int>(cfg.num_batches.value());
        std::vector<double> gathered_batch_times(mpi_rank == 0 ? total_batches : 0);
        // Gather batch times
        MPI_Gather(batch_times.data(), batches_per_process, MPI_DOUBLE,
                   gathered_batch_times.data(), batches_per_process, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        const int logits_per_rank = batches_per_process * static_cast<int>(cfg.batch_size) * static_cast<int>(model.num_classes());
        const int preds_per_rank  = batches_per_process * static_cast<int>(cfg.batch_size);
        std::vector<float> gathered_logits(mpi_rank == 0 ? static_cast<std::size_t>(mpi_size) * logits_per_rank : 0);
        std::vector<std::int64_t> gathered_preds(mpi_rank == 0 ? static_cast<std::size_t>(mpi_size) * preds_per_rank : 0);
        // Gather logits and preds
        MPI_Gather(all_logits.data(), logits_per_rank, MPI_FLOAT,
                   gathered_logits.data(), logits_per_rank, MPI_FLOAT,
                   0, MPI_COMM_WORLD);
        // Gather predicted class ids
        MPI_Gather(all_preds.data(), preds_per_rank, MPI_INT64_T,
                   gathered_preds.data(), preds_per_rank, MPI_INT64_T,
                   0, MPI_COMM_WORLD);
#else
        const int total_batches = batches_per_process;
        const std::vector<double>& gathered_batch_times = batch_times;
        const std::vector<float>& gathered_logits = all_logits;
        const std::vector<std::int64_t>& gathered_preds = all_preds;
#endif

        if (mpi_rank == 0) {
            // average latency
            const double mean_ms = std::accumulate(gathered_batch_times.begin(), gathered_batch_times.end(), 0.0) / total_batches;

            // standard deviation
            double variance = 0.0;
            for (const double t : gathered_batch_times)
            {
                variance += (t - mean_ms) * (t - mean_ms);
            }
            const double std_ms = std::sqrt(variance / total_batches);

#ifdef USE_MPI
            // Throughput = all ranks process in parallel, wall time = mean latency per batch
            const double throughput = static_cast<double>(mpi_size * cfg.batch_size) / (mean_ms / 1000.0);
#else
            const double throughput = cfg.batch_size / (mean_ms / 1000.0);
#endif

            std::cout << "================================================================================\n";
#ifdef USE_MPI
            std::cout << "  Benchmark Results (MPI, " << mpi_size << " ranks)\n";
#else
            std::cout << "  Benchmark Results\n";
#endif
            profile_data.print_timestamps();
            std::cout << "================================================================================\n";
            std::cout << "  Mean latency : " << mean_ms << " ms\n";
            std::cout << "  Std dev      : " << std_ms << " ms\n";
            std::cout << "  Throughput   : " << throughput << " samples/sec\n";
            std::cout << "================================================================================\n";

            // write results to files
            write_float32_file(cfg.logits_out, gathered_logits);
            write_int64_file(cfg.preds_out, gathered_preds);

            std::cout << "Wrote logits to: " << cfg.logits_out << "\n";
            std::cout << "Wrote preds  to: " << cfg.preds_out << "\n";
            if (!cfg.quiet) {
                const auto &class_names = model.class_names();
                std::cout << "\nFirst 10 predictions:\n";
                for (std::size_t i = 0; i < std::min<std::size_t>(25, gathered_preds.size()); ++i)
                {
                    const auto cls = static_cast<std::size_t>(gathered_preds[i]);
                    std::cout << "  sample " << i << ": class_id=" << gathered_preds[i];
                    if (cls < class_names.size())
                    {
                        std::cout << " (" << class_names[cls] << ")";
                    }
                    std::cout << "\n";
                }
            }
        }

#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "ERROR: " << ex.what() << "\n";
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
}
