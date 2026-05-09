cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MLPInfer LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Print the compiler being used (Helpful on Dardel to confirm 'CC' is active)
message(STATUS "C++ Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Compiler Path: ${CMAKE_CXX_COMPILER}")

# Default to Release mode for optimization (-O3 equivalent)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Define the executable and its source files
add_executable(mlp_infer
    src/main.cpp
    src/MLPModel.cpp
)

# Tell the compiler where to find your header files
target_include_directories(mlp_infer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find and link MPI
find_package(MPI REQUIRED)        
target_link_libraries(mlp_infer PRIVATE MPI::MPI_CXX)