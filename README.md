# HPC MLP Inference

High-performance C++ inference for a multilayer perceptron (MLP) classifier on CIFAR-10 embeddings.

This project implements and benchmarks several increasingly optimized versions of MLP inference, starting from a scalar CPU baseline and extending it with custom memory management, SIMD vectorization, OpenMP shared-memory parallelism, MPI distributed-memory parallelism, and GPU offloading experiments.

The model takes **512-dimensional `float32` image embeddings** and predicts one of **10 CIFAR-10 classes**. The network is a fully connected MLP with layer dimensions:

```text
512 -> 4096 -> 4096 -> 4096 -> 10
```

ReLU is applied after each hidden layer. The input embeddings are produced by a ResNet18 encoder, so this repository performs inference on embeddings, not directly on raw CIFAR-10 images.

---

## Highlights

* C++ implementation of MLP forward inference
* Flat row-major matrix representation for dense tensors
* Non-owning `FlatMatrixView` to avoid unnecessary matrix copies
* Custom allocator / memory-pool experiments to reduce allocation and initialization overhead
* AVX2/FMA-friendly matrix multiplication kernels
* OpenMP row-based and tiled parallel matrix multiplication variants
* MPI data-parallel inference across ranks/nodes
* GPU offloading experiments using cuBLAS SGEMM and CUDA kernels
* Benchmark binaries for comparing scalar, auto-vectorized, AVX2, native, OpenMP, and tiled implementations
* Unit-test binaries for validating matrix algorithms under different compile-time configurations

---

## Repository Layout

```text
include/
├── CLI11.hpp
├── FlatMatrix.hpp
├── FlatMatrixView.hpp
├── FreeListAllocator.hpp
├── FreeListMemoryPool.hpp
├── json.hpp
├── LoggingAllocator.hpp
├── MatrixAlgorithms.hpp
├── MatrixConcepts.hpp
├── MLPModel.hpp
└── utils.hpp

src/
├── main.cpp
├── MLPModel.cpp
├── flat_matrix_benchmark.cpp
├── matrix_algorithms_tests.cpp
├── avx2_fma_throughput_benchmark.cpp
└── avx2_fma_throughput_benchmark

scripts/
├── run_perf_tests.sh
├── bin_to_text.py
├── plot_perf_metric_comparison_hardcoded.py
├── plot_sparse_dense_view.py
└── benchmark / optimization plots

results/
├── benchmark summaries
├── scaling results
├── perf outputs
├── disassembly files
└── hpc_mlp_architecture.png

cifar10_embedding_mlp/
├── best_mlp.pt
├── train_embeddings.pt
├── test_embeddings.pt
└── export/
    ├── layer_00_weight.bin
    ├── layer_00_bias.bin
    ├── layer_01_weight.bin
    ├── layer_01_bias.bin
    ├── layer_02_weight.bin
    ├── layer_02_bias.bin
    ├── layer_03_weight.bin
    ├── layer_03_bias.bin
    ├── metadata.json
    └── test input / reference output files
```

---

## Model Export

The C++ inference code expects the trained model weights and test embeddings to be exported as binary files.

Create a Python environment:

```bash
conda create -n mlp-cifar python=3.11 -y
conda activate mlp-cifar

pip install torch torchvision torchaudio jupyter matplotlib numpy tqdm ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name mlp-cifar --display-name "Python (mlp-cifar)"
jupyter notebook
```

After running the model generation/export notebook, the export directory should contain files similar to:

```text
cifar10_embedding_mlp/export/
├── layer_00_bias.bin
├── layer_00_weight.bin
├── layer_01_bias.bin
├── layer_01_weight.bin
├── layer_02_bias.bin
├── layer_02_weight.bin
├── layer_03_bias.bin
├── layer_03_weight.bin
├── metadata.json
├── test_input_embeddings_all.bin
├── test_labels_int64_all.bin
├── test_output_logits_all.bin
├── test_output_pred_int64_all.bin
└── test_shapes_all.json
```

In some local experiments, the test inputs are organized into manually created subdirectories such as:

```text
cifar10_embedding_mlp/export/test_input_all/
cifar10_embedding_mlp/export/test_input_small/
```

Adjust the input paths in the run commands depending on where your exported test files are located.

---

## Building

### Recommended CMake Build

```bash
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

The CMake project uses **C++20** and enables `-O3` by default for optimized builds.

### Manual Minimal Build

For a simple local build of the main inference executable:

```bash
g++ -O3 -std=c++20 -Iinclude \
    src/main.cpp src/MLPModel.cpp \
    -o mlp_infer
```

For AVX2/FMA optimization on x86-64:

```bash
g++ -O3 -std=c++20 -Iinclude \
    -mavx2 -mfma -ffast-math \
    src/main.cpp src/MLPModel.cpp \
    -o mlp_infer_avx2
```

### Building on Dardel

On Dardel, switch to the GNU programming environment before building:

```bash
module swap PrgEnv-cray PrgEnv-gnu
```

Then build with CMake:

```bash
mkdir -p build
cd build

CXX=g++ cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

To confirm the compiler:

```bash
g++ --version
```

### OpenMP Notes

The CMake configuration automatically searches for OpenMP. If OpenMP is found, OpenMP-enabled inference, benchmark, and test binaries are built.

On macOS with AppleClang, install OpenMP first:

```bash
brew install libomp
```

The CMake file contains special handling for AppleClang and Homebrew `libomp`.

---

## Running Inference

### Named-Argument CLI

```bash
./build/mlp_infer \
    --export-dir cifar10_embedding_mlp/export \
    --input cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin \
    --logits-out cpp_output_logits_all.bin \
    --preds-out cpp_output_preds_all.bin \
    --batch-size 64
```

### Positional-Argument Form

Some versions of the executable also support positional arguments:

```bash
./build/mlp_infer \
    cifar10_embedding_mlp/export \
    cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin \
    cpp_output_logits_all.bin \
    cpp_output_preds_all.bin
```

From inside the `build/` directory, use paths relative to `build/`:

```bash
./mlp_infer \
    ../cifar10_embedding_mlp/export \
    ../cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin \
    ../cpp_output_logits_all.bin \
    ../cpp_output_preds_all.bin
```

The executable writes:

```text
cpp_output_logits_all.bin    # float32 logits
cpp_output_preds_all.bin     # int64 predicted class IDs
```

---

## Running with OpenMP

Use the OpenMP-optimized binary when available:

```bash
export OMP_NUM_THREADS=32
export OMP_PLACES=cores

./build/mlp_infer_avx2_max_openmp \
    --export-dir cifar10_embedding_mlp/export \
    --input cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin \
    --logits-out cpp_output_logits_all.bin \
    --preds-out cpp_output_preds_all.bin \
    --batch-size 128
```

You can also experiment with thread placement:

```bash
export OMP_PROC_BIND=close
# or
export OMP_PROC_BIND=spread
```

The project benchmarks showed that thread placement can be machine-dependent: binding helped on some school-cluster runs but did not consistently improve performance on Dardel.

---

## Running with MPI

The MPI implementation uses data parallelism: rank 0 distributes input embeddings across ranks, each rank performs inference on its assigned samples, and the results are gathered back.

Example:

```bash
mpirun -n 4 ./build/mlp_infer \
    --export-dir cifar10_embedding_mlp/export \
    --input cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin \
    --logits-out cpp_output_logits_all.bin \
    --preds-out cpp_output_preds_all.bin \
    --batch-size 64
```

For hybrid MPI + OpenMP runs:

```bash
export OMP_NUM_THREADS=8
export OMP_PLACES=cores

mpirun -n 16 ./build/mlp_infer_avx2_max_openmp \
    --export-dir cifar10_embedding_mlp/export \
    --input cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin \
    --logits-out cpp_output_logits_all.bin \
    --preds-out cpp_output_preds_all.bin \
    --batch-size 128
```

If your MPI implementation requires compiling through an MPI wrapper, configure CMake with:

```bash
CXX=mpicxx cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

---

## CMake Targets

The CMake file builds several executables for inference, benchmarking, and testing different optimization levels.

### Main Inference Targets

| Target                          | Purpose                                                                                                     |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `mlp_infer`                     | Default inference executable.                                                                               |
| `mlp_infer_no_opt`              | Scalar baseline with compiler vectorization and AVX disabled. Useful as the performance floor.              |
| `mlp_infer_native`              | Uses `-march=native` and `-ffast-math` so the compiler can tune for the host CPU.                           |
| `mlp_infer_avx2_max`            | Uses AVX2/FMA flags and enables the DOT4 matrix multiplication kernel.                                      |
| `mlp_infer_custom_alloc`        | AVX2/FMA + DOT4 + disabled default initialization for output buffers.                                       |
| `mlp_infer_custom_alloc_no_opt` | Scalar/no-vectorization version with disabled default initialization.                                       |
| `mlp_infer_avx2_max_openmp`     | AVX2/FMA + DOT4 + custom allocation behavior + tiled OpenMP parallelization. Built only if OpenMP is found. |

### Matrix Benchmark Targets

These targets are based on `src/flat_matrix_benchmark.cpp`.

| Target                                          | Purpose                                                                                |
| ----------------------------------------------- | -------------------------------------------------------------------------------------- |
| `flat_matrix_benchmark_0_scalar`                | Pure scalar baseline.                                                                  |
| `flat_matrix_benchmark_1_fast_math`             | Scalar code with `-ffast-math`.                                                        |
| `flat_matrix_benchmark_2_auto_vec`              | Compiler auto-vectorization with default target flags.                                 |
| `flat_matrix_benchmark_3_avx2_fma`              | AVX2 + FMA benchmark.                                                                  |
| `flat_matrix_benchmark_4_native`                | Native CPU tuning using `-march=native`.                                               |
| `flat_matrix_benchmark_5_avx2_dot4`             | Explicit DOT4-style kernel intended to expose more SIMD-friendly work to the compiler. |
| `flat_matrix_benchmark_5_avx2_kernel_2_4`       | AVX2 kernel variant computing a 2 × 4 output tile.                                     |
| `flat_matrix_benchmark_5_avx2_kernel_3_4`       | AVX2 kernel variant computing a 3 × 4 output tile.                                     |
| `flat_matrix_benchmark_5_avx2_kernel_6_4`       | AVX2 kernel variant computing a 6 × 4 output tile.                                     |
| `flat_matrix_benchmark_6_avx_dot4_openmp`       | DOT4 benchmark with OpenMP row-based parallelization.                                  |
| `flat_matrix_benchmark_7_avx_dot4_openmp_tiled` | DOT4 benchmark with tiled OpenMP parallelization.                                      |

Example:

```bash
./build/flat_matrix_benchmark_5_avx2_dot4
./build/flat_matrix_benchmark_7_avx_dot4_openmp_tiled
```

### Matrix Algorithm Test Targets

These targets are based on `src/matrix_algorithms_tests.cpp`.

| Target                                            | Purpose                                 |
| ------------------------------------------------- | --------------------------------------- |
| `matrix_algorithms_tests_1_fast_math`             | Tests with scalar fast-math flags.      |
| `matrix_algorithms_tests_2_auto_vec`              | Tests with compiler auto-vectorization. |
| `matrix_algorithms_tests_3_avx2_fma`              | Tests with AVX2/FMA flags.              |
| `matrix_algorithms_tests_4_native`                | Tests with native CPU tuning.           |
| `matrix_algorithms_tests_5_avx2_dot4`             | Tests the DOT4 kernel configuration.    |
| `matrix_algorithms_tests_5_avx2_kernel_2_4`       | Tests the 2 × 4 kernel configuration.   |
| `matrix_algorithms_tests_5_avx2_kernel_3_4`       | Tests the 3 × 4 kernel configuration.   |
| `matrix_algorithms_tests_5_avx2_kernel_6_4`       | Tests the 6 × 4 kernel configuration.   |
| `matrix_algorithms_tests_6_avx_dot4_openmp`       | Tests DOT4 with OpenMP.                 |
| `matrix_algorithms_tests_7_avx_dot4_openmp_tiled` | Tests DOT4 with tiled OpenMP.           |

Example:

```bash
./build/matrix_algorithms_tests_5_avx2_dot4
./build/matrix_algorithms_tests_7_avx_dot4_openmp_tiled
```

### AVX2 FMA Throughput Benchmark

The source file:

```text
src/avx2_fma_throughput_benchmark.cpp
```

measures approximate maximum AVX2 FMA throughput on the current CPU.

If it is not built by CMake in your local version, compile it manually:

```bash
g++ -O3 -std=c++20 -mavx2 -mfma -ffast-math \
    src/avx2_fma_throughput_benchmark.cpp \
    -o avx2_fma_throughput_benchmark
```

Run:

```bash
./avx2_fma_throughput_benchmark
```

---

## Optimization Summary

### Memory Management

The inference pipeline uses flat row-major matrices to improve spatial locality for row-wise traversal and keep indexing simple.

Two important memory optimizations are used:

1. **`FlatMatrixView`**
   A non-owning matrix view used to avoid copying large immutable operands such as model weights.

2. **Custom allocation / disabled initialization**
   Output matrices are overwritten during matrix multiplication, so default zero/value initialization is unnecessary. The custom allocator configuration avoids this initialization overhead.

### SIMD / AVX2

The SIMD-optimized kernels target AVX2 and FMA-capable CPUs.

The optimized matrix multiplication variants compute small output tiles at a time, such as 2 × 4, 3 × 4, or 6 × 4 tiles. This improves instruction-level parallelism and data reuse. The best-performing variants are designed so the compiler can generate 256-bit AVX2 instructions, processing eight `float32` values per vector instruction.

A key tradeoff is register pressure. Larger tile sizes expose more parallel work, but they also require more vector accumulators. If too many accumulators are needed, the compiler may spill registers to the stack, reducing performance.

### OpenMP

OpenMP parallelizes the matrix multiplication, which dominates runtime.

Two strategies are benchmarked:

1. **Row-based splitting**
   Each thread receives groups of rows. This is simple but can create too few work units for small batch sizes.

2. **Tiled splitting**
   Work is split into smaller 4 × 4 output tiles. This creates more independent tasks and keeps more threads busy, especially in strong-scaling experiments.

OpenMP performance depends heavily on batch size, thread count, cache behavior, and memory bandwidth. For small fixed batches, efficiency drops at high thread counts because there is not enough independent work. In weak scaling, where batch size grows with the number of threads, scaling improves but can still become memory-bandwidth limited.

### MPI

MPI extends inference across processes or nodes.

The implementation follows a data-parallel structure:

1. Rank 0 loads or coordinates the input.
2. Inputs are scattered across MPI ranks.
3. Each rank runs local MLP inference independently.
4. Logits and predictions are gathered back to rank 0.

There is no inter-rank communication during the hot inference loop. Communication happens mainly through broadcast, scatter, and gather operations outside the main computation.

The project results show that MPI scales much better across multiple nodes than on a single shared-memory node. On a single node, multiple ranks compete for the same memory bandwidth and cache resources, while multi-node runs benefit from separate memory bandwidth per node.

### GPU Offloading

The report also evaluates a GPU backend.

The GPU version offloads matrix multiplications to cuBLAS SGEMM and fuses bias addition and ReLU into custom CUDA kernels. Weights stay resident on the GPU, and only the input batch is transferred for each inference call.

The first batch includes CUDA and cuBLAS warmup overhead, but later batches are much faster. In the reported experiments, a single NVIDIA H100 achieved very high throughput and low latency compared with CPU configurations, although the CPU and GPU results should be interpreted carefully because they were measured on different hardware configurations.

---

## Benchmark Results Summary

The project demonstrates large speedups from each level of optimization:

* `-march=native` gives a large improvement over scalar code because the compiler can use vectorized instructions.
* AVX2/FMA kernels significantly improve single-core matrix multiplication throughput.
* OpenMP improves performance on a single node, but scaling eventually becomes limited by available work and memory bandwidth.
* MPI scales well on multi-node systems because each node provides additional memory bandwidth.
* GPU offloading provides very high throughput for large GEMM-heavy workloads.

Selected reported results:

| Configuration                                      |              Result |
| -------------------------------------------------- | ------------------: |
| Local scalar matrix multiplication                 |          233.017 ms |
| Local native matrix multiplication                 |           57.728 ms |
| Local AVX2 max matrix multiplication               |           12.570 ms |
| Dardel AVX2 max matrix multiplication              |           14.489 ms |
| Dardel OpenMP + MPI, 64 MPI ranks × 64 OMP threads | 138,186 samples/sec |
| School cluster H100 GPU, warm steady state         |  55,172 samples/sec |

The best CPU result used 64 compute nodes, while the GPU result used a different system with one H100 GPU, so these results should not be treated as a direct hardware-to-hardware comparison.

---

## Running Benchmarks

Run a specific benchmark binary:

```bash
./build/flat_matrix_benchmark_0_scalar
./build/flat_matrix_benchmark_4_native
./build/flat_matrix_benchmark_5_avx2_dot4
./build/flat_matrix_benchmark_7_avx_dot4_openmp_tiled
```

Run the performance script:

```bash
bash scripts/run_perf_tests.sh
```

Plotting scripts are located in `scripts/`:

```bash
python scripts/plot_perf_metric_comparison_hardcoded.py
python scripts/plot_sparse_dense_view.py
```

Generated and saved benchmark outputs are stored in `results/`.

---

## Verifying Correctness

Run the matrix algorithm tests:

```bash
./build/matrix_algorithms_tests_2_auto_vec
./build/matrix_algorithms_tests_5_avx2_dot4
```

For OpenMP-enabled builds:

```bash
./build/matrix_algorithms_tests_7_avx_dot4_openmp_tiled
```

For full inference correctness, compare the generated C++ output files against the exported PyTorch reference outputs:

```text
test_output_logits_all.bin
test_output_pred_int64_all.bin
```

The C++ executable produces:

```text
cpp_output_logits_all.bin
cpp_output_preds_all.bin
```

---

## Input and Output File Format

The exported model directory contains:

```text
metadata.json
layer_00_weight.bin
layer_00_bias.bin
layer_01_weight.bin
layer_01_bias.bin
layer_02_weight.bin
layer_02_bias.bin
layer_03_weight.bin
layer_03_bias.bin
```

The test input contains `float32` embeddings:

```text
test_input_embeddings_all.bin
```

The reference labels and outputs are:

```text
test_labels_int64_all.bin
test_output_logits_all.bin
test_output_pred_int64_all.bin
test_shapes_all.json
```

The C++ inference program writes:

```text
cpp_output_logits_all.bin
cpp_output_preds_all.bin
```

---

## Troubleshooting

### Illegal instruction on older CPUs

The AVX2 binaries require AVX2/FMA support. If the program crashes with an illegal instruction, use one of the scalar or native targets instead:

```bash
./build/mlp_infer_no_opt
./build/mlp_infer_native
```

### OpenMP target not built

If `mlp_infer_avx2_max_openmp` is missing, CMake did not find OpenMP. Install OpenMP and rerun CMake.

On macOS:

```bash
brew install libomp
```

Then rebuild:

```bash
rm -rf build
mkdir build && cd build
cmake ..
make -j
```

### Wrong input path

If the executable cannot open the input file, check whether your test inputs are directly in:

```text
cifar10_embedding_mlp/export/
```

or in:

```text
cifar10_embedding_mlp/export/test_input_all/
```

Update the `--input` path accordingly.

### MPI run does not start

Make sure MPI is available on the system:

```bash
mpirun --version
```

If the code requires MPI headers during compilation, build with an MPI compiler wrapper:

```bash
CXX=mpicxx cmake ..
make -j
```

---

## Future Work

Potential directions for further optimization:

* Improve matrix storage layout to reduce cache misses
* Exploit ReLU-induced sparsity in hidden activations
* Add sparse-dense matrix multiplication kernels
* Improve NUMA-aware memory placement for multi-socket systems
* Add a first-class CUDA/CMake target if the GPU backend is included in the repository
* Add automated correctness checks comparing C++ outputs against PyTorch reference outputs
