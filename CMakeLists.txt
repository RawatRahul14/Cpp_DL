cmake_minimum_required(VERSION 3.20)

project(NeuralNetCppWineQuality LANGUAGES CXX)

# ==============================================================================
# C++ Standard Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# External Dependencies
# ==============================================================================
# 1. LibTorch (C++ frontend for PyTorch)
find_package(Torch REQUIRED)

# 2. Python Interpreter & Development Headers (Python.h / python311.lib)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 3. pybind11 (C++20 <-> Python C-API bridge)
find_package(pybind11 CONFIG REQUIRED)

# ==============================================================================
# Target: wine_engine (Python C-Extension / .pyd on Windows)
# ==============================================================================
pybind11_add_module(wine_engine
    src/bindings.cpp
    src/wine_model.cpp
)

# Expose local header directory
target_include_directories(wine_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against LibTorch shared libraries (torch_cpu, c10, etc.)
target_link_libraries(wine_engine PRIVATE
    "${TORCH_LIBRARIES}"
)

# Append required LibTorch compiler flags (ABI & macro definitions)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ==============================================================================
# Compiler-Specific Warning & Optimization Flags
# ==============================================================================
if(MSVC)
    # /W4     : Enable high warning level for our own project code
    # /wd4267 : Suppress 'size_t' to 'int' narrowing warnings originating inside LibTorch headers
    # /wd4702 : Suppress unreachable code warnings triggered by template instantiations
    target_compile_options(wine_engine PRIVATE /W4 /wd4267 /wd4702)
endif()