cmake_minimum_required(VERSION 3.25)
project(riemannian_gaussian_sampler VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Backend forwarding – mirrors the isomorphism library options.
#option(USE_MLX  "Build with Apple Silicon (Metal/MLX) backend" OFF)
#option(USE_SYCL "Build with PC/Intel/NVIDIA (SYCL/oneMKL) backend" OFF)

#if (USE_MLX AND USE_SYCL)
#    message(FATAL_ERROR "Choose either USE_MLX or USE_SYCL, not both.")
#elseif (NOT USE_MLX AND NOT USE_SYCL)
#    message(FATAL_ERROR "No backend selected. Use -DUSE_MLX=ON or -DUSE_SYCL=ON.")
#endif ()

if (APPLE)
    # Provide a hint to CMake for Homebrew's libomp location
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
endif()
find_package(OpenMP REQUIRED)

# ==============================================================================
# Locate the isomorphism library
# ==============================================================================
# Preferred: set ISOMORPHISM_DIR to the root of the isomorphism source tree.
# CMake will then configure it as a sub-project so both libraries are built
# together and changes to isomorphism are always reflected here.
set(ISOMORPHISM_DIR "" CACHE PATH "Path to the isomorphism source directory")

if (ISOMORPHISM_DIR STREQUAL "")
    # Fallback: attempt sibling directory heuristic.
    get_filename_component(_parent "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    set(ISOMORPHISM_DIR "${_parent}/isomorphism")
endif ()

if (EXISTS "${ISOMORPHISM_DIR}/CMakeLists.txt")
    message(STATUS "Sampler: using isomorphism source at ${ISOMORPHISM_DIR}")
    add_subdirectory(${ISOMORPHISM_DIR} isomorphism_build)
else ()
    # Already installed – use find_package.
    find_package(isomorphism REQUIRED)
endif ()

# ==============================================================================
# Sampler library
# ==============================================================================
add_library(sampler STATIC
    src/thread_config.cpp
    src/angle_sampler_hmc.cpp
    src/principal_angle_sampler.cpp
    src/stiefel_gaussian_sampler.cpp
    src/so_angle_sampler.cpp
    src/so_gaussian_sampler.cpp
)

target_include_directories(sampler
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(sampler PUBLIC isomorphism OpenMP::OpenMP_CXX)

# ==============================================================================
# Optional demo executable
# ==============================================================================
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    add_executable(sampler_demo main.cpp)
    target_link_libraries(sampler_demo PRIVATE sampler)
endif ()

# ==============================================================================
# Build-type flags
# ==============================================================================
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(sampler PRIVATE -O3 -ffast-math -march=native)
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Mirror the sanitizer flags that isomorphism adds in Debug mode so that
    # the sampler and test executables link against the same sanitizer runtime.
    add_compile_options(-fsanitize=address -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=undefined)
endif ()

# ==============================================================================
# Tests
# ==============================================================================
add_subdirectory(tests)
