cmake_minimum_required(VERSION 3.28)
project(mslab VERSION 0.1.0 LANGUAGES CXX)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Find dependencies
find_package(pybind11 REQUIRED)

# Core library (all your engine code goes here)
add_library(mslab_core
    cpp/src/book.cpp
)
target_include_directories(mslab_core PUBLIC cpp/include)

# Python bindings
pybind11_add_module(mslab_bindings cpp/src/bindings.cpp)
target_link_libraries(mslab_bindings PRIVATE mslab_core)

# Enable testing
enable_testing()
add_subdirectory(cpp/tests)
add_subdirectory(cpp/bench)