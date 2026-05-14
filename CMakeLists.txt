cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MLPInfer LANGUAGES CXX)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Print the compiler being used (Helpful on Dardel to confirm 'CC' is active)
message(STATUS "C++ Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Compiler Path: ${CMAKE_CXX_COMPILER}")

# Default to Release mode for optimization (-O3 equivalent)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3)

if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    link_libraries(OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found via find_package, trying -fopenmp directly")
    add_compile_options(-fopenmp)
    link_libraries(-fopenmp)
endif()

# x86-specific flags only apply on x86_64. On ARM (Apple Silicon) AVX doesn't exist
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(FLAGS_NO_VEC   -fno-tree-vectorize -mno-avx -mno-avx2)
    set(FLAGS_AVX2_MAX -O3 -mavx2 -mfma -ffast-math)
    set(FLAGS_NATIVE   -O3 -march=native -ffast-math)
else()
    set(FLAGS_NO_VEC   -fno-tree-vectorize)
    set(FLAGS_AVX2_MAX -O3 -march=native -ffast-math)
    set(FLAGS_NATIVE   -O3 -march=native -ffast-math)
endif()

include_directories(include)

# Define common source files for all MLP targets
set(MLP_SOURCES
        src/main.cpp
        src/MLPModel.cpp
)

add_executable(mlp_infer ${MLP_SOURCES})

# --- 1. The "Floor": No Optimization ---
# This is your scalar baseline. We explicitly turn off the "smart" features.
add_executable(mlp_infer_no_opt ${MLP_SOURCES})
target_compile_options(mlp_infer_no_opt PRIVATE ${FLAGS_NO_VEC})

# --- 3. The "Ceiling": AVX2 Max ---
# This uses everything the compiler has: native tuning and OpenMP SIMD hints.
add_executable(mlp_infer_avx2_max ${MLP_SOURCES})
target_compile_options(mlp_infer_avx2_max PRIVATE ${FLAGS_AVX2_MAX})
# We enable the DOT4 unrolled logic for the MAX version
target_compile_definitions(mlp_infer_avx2_max PRIVATE
        FLATMATRIX_USE_DOT4=1
)


set(FLAT_MATRIX_BENCHMARK_SOURCE src/flat_matrix_benchmark.cpp)

# --- STEP 0: Pure Scalar (The Baseline) ---
# No vectorization, no hardware tricks. Just standard math logic.
add_executable(flat_matrix_benchmark_0_scalar ${FLAT_MATRIX_BENCHMARK_SOURCE})
target_compile_options(flat_matrix_benchmark_0_scalar PRIVATE -O3 ${FLAGS_NO_VEC})

# --- STEP 1: Scalar + Fast Math ---
# Still no vectorization, but we allow the compiler to reorder math for better pipelining.
add_executable(flat_matrix_benchmark_1_fast_math ${FLAT_MATRIX_BENCHMARK_SOURCE})
target_compile_options(flat_matrix_benchmark_1_fast_math PRIVATE
        -O3 -fno-tree-vectorize -ffast-math)

# --- STEP 2: Auto-Vectorization (Generic x86-64) ---
# We let the compiler use SIMD, but only standard SSE (128-bit) that works on any 64-bit PC.
add_executable(flat_matrix_benchmark_2_auto_vec ${FLAT_MATRIX_BENCHMARK_SOURCE})
target_compile_options(flat_matrix_benchmark_2_auto_vec PRIVATE
        -O3)

# --- STEP 3: AVX2 + FMA (The Modern Standard) ---
# Moving to 256-bit wide registers and fused multiply-add.
add_executable(flat_matrix_benchmark_3_avx2_fma ${FLAT_MATRIX_BENCHMARK_SOURCE})
target_compile_options(flat_matrix_benchmark_3_avx2_fma PRIVATE ${FLAGS_AVX2_MAX})

# --- STEP 4: Native Tuning ---
# Let the compiler use every specific trick your CPU has (like avx_vnni).
add_executable(flat_matrix_benchmark_4_native ${FLAT_MATRIX_BENCHMARK_SOURCE})
target_compile_options(flat_matrix_benchmark_4_native PRIVATE ${FLAGS_NATIVE})

# --- STEP 5: Explicit code so that compiler can vectorize as much as possible ---
add_executable(flat_matrix_benchmark_5_avx2_dot4 ${FLAT_MATRIX_BENCHMARK_SOURCE})
target_compile_options(flat_matrix_benchmark_5_avx2_dot4 PRIVATE ${FLAGS_AVX2_MAX})
target_compile_definitions(flat_matrix_benchmark_5_avx2_dot4 PRIVATE FLATMATRIX_USE_DOT4=1)

# Tell the compiler where to find your header files
target_include_directories(mlp_infer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
