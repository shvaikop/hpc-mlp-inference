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

include_directories(include)

# Define the executable and its source files
add_executable(mlp_infer
    src/main.cpp
    src/MLPModel.cpp
)

add_compile_options(-O3)

add_executable(flat_matrix_benchmark_avx2
        src/flat_matrix_benchmark.cpp
)

target_compile_options(flat_matrix_benchmark_avx2 PRIVATE
#        -march=native
        -mavx2
        -mfma
        -ffast-math
#        -fopt-info-vec-optimized
#        -fopt-info-vec-missed
)

add_executable(flat_matrix_benchmark_native
        src/flat_matrix_benchmark.cpp
)

target_compile_options(flat_matrix_benchmark_native PRIVATE
        -march=native
        -mfma
        -ffast-math
#        -fopt-info-vec-optimized
#        -fopt-info-vec-missed
)

add_executable(flat_matrix_benchmark_dot4
        src/flat_matrix_benchmark.cpp
)

target_compile_options(flat_matrix_benchmark_dot4 PRIVATE
        -O3
        -mavx2
        -mfma
        -ffast-math
        -fopt-info-vec-optimized
        -fopt-info-vec-missed
)

target_compile_definitions(flat_matrix_benchmark_dot4 PRIVATE
        FLATMATRIX_USE_DOT4=1
)

add_executable(flat_matrix_benchmark_dot4_openmp_simd
        src/flat_matrix_benchmark.cpp
)

target_compile_options(flat_matrix_benchmark_dot4_openmp_simd PRIVATE
        -O3
        -mavx2
        -mfma
        -ffast-math
        -fopenmp-simd
        -fopt-info-vec-optimized
        -fopt-info-vec-missed
)

target_compile_definitions(flat_matrix_benchmark_dot4_openmp_simd PRIVATE
        FLATMATRIX_USE_DOT4=1
)

# Tell the compiler where to find your header files
target_include_directories(mlp_infer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
