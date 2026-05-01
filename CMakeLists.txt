cmake_minimum_required(VERSION 3.25)
project(riemannian_gaussian_sampler VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ==============================================================================
# OpenMP
# ==============================================================================
if (APPLE)
    # Provide a hint to CMake for Homebrew's libomp location
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
endif()
find_package(OpenMP REQUIRED)

# ==============================================================================
# Locate the isomorphism library
# ==============================================================================
# Development mode: set ISOMORPHISM_DIR to the root of the isomorphism source
# tree and CMake will configure it as a sub-project.
# Install mode (brew): leave unset — find_package picks up the installed copy.
set(ISOMORPHISM_DIR "" CACHE PATH "Path to the isomorphism source directory (development only)")

if (ISOMORPHISM_DIR STREQUAL "")
    get_filename_component(_parent "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    set(ISOMORPHISM_DIR "${_parent}/isomorphism")
endif ()

if (EXISTS "${ISOMORPHISM_DIR}/CMakeLists.txt")
    message(STATUS "Sampler: using isomorphism source at ${ISOMORPHISM_DIR}")
    add_subdirectory(${ISOMORPHISM_DIR} isomorphism_build)
    # add_subdirectory creates the plain 'isomorphism' alias directly.
    set(_iso_link_target isomorphism)
else ()
    find_package(isomorphism REQUIRED)
    # find_package only exports namespaced targets (isomorphism::mlx, etc.)
    # and sets isomorphism_DEFAULT_TARGET to whichever backend was installed.
    if (NOT isomorphism_DEFAULT_TARGET)
        message(FATAL_ERROR
            "find_package(isomorphism) succeeded but isomorphism_DEFAULT_TARGET "
            "is not set. Re-install isomorphism with a backend enabled "
            "(-DUSE_MLX=ON, -DUSE_EIGEN=ON, or -DUSE_TORCH=ON).")
    endif ()
    set(_iso_link_target ${isomorphism_DEFAULT_TARGET})
    message(STATUS "Sampler: linking against installed ${_iso_link_target}")
endif ()

# ==============================================================================
# Sampler library
# ==============================================================================
add_library(sampler STATIC
    # Shared infrastructure
    src/thread_config.cpp
    src/angle_sampler_hmc.cpp
    # Isotropic samplers (SO(d) + Stiefel)
    src/isotropic/principal_angle_sampler.cpp
    src/isotropic/stiefel_gaussian_sampler.cpp
    src/isotropic/so_angle_sampler.cpp
    src/isotropic/so_gaussian_sampler.cpp
    src/isotropic/so_heterogeneous_gaussian_sampler.cpp
    src/isotropic/stiefel_heterogeneous_gaussian_sampler.cpp
    # Anisotropic samplers
    src/anisotropic/so_anisotropic_angle_sampler.cpp
    src/anisotropic/so_hypersphere_sampler_cpu.cpp
    src/anisotropic/so_hypersphere_sampler_cpu_spatial.cpp
    src/anisotropic/so_hypersphere_sampler_gpu.cpp
    src/anisotropic/so_anisotropic_gaussian_sampler.cpp
    src/anisotropic/so_anisotropic_spatial_gaussian_sampler.cpp
    src/anisotropic/so_anisotropic_heterogeneous_gaussian_sampler.cpp
)

# sampler:: namespace alias so consumers can write target_link_libraries(... sampler::sampler)
add_library(sampler::sampler ALIAS sampler)

target_include_directories(sampler
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(sampler PUBLIC ${_iso_link_target} OpenMP::OpenMP_CXX)

# ==============================================================================
# Build-type flags
# ==============================================================================
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    # -ffast-math is deliberately excluded. The HMC density evaluator relies on:
    #   - std::isfinite() for trajectory validity (broken by -ffinite-math-only)
    #   - -infinity as a Weyl-chamber rejection sentinel
    #   - exact division and log for correct Metropolis-Hastings acceptance
    #   - associative floating-point accumulation order in gradient sums
    # All of these are invalidated by one or more sub-flags of -ffast-math.
    # The heavy computation is done by the isomorphism GPU backend, which applies
    # its own fast-math settings. The sampler gains little from it here.
    target_compile_options(sampler PRIVATE -O3 -march=native)
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    #add_compile_options(-fsanitize=address -fsanitize=undefined)
    #add_link_options(-fsanitize=address -fsanitize=undefined)
endif ()

# ==============================================================================
# Optional demo executable (never installed)
# ==============================================================================
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    add_executable(sampler_demo main.cpp)
    target_link_libraries(sampler_demo PRIVATE sampler)
endif ()

# ==============================================================================
# Tests (only when this is the top-level project, not when consumed via
# add_subdirectory or during a brew install)
# ==============================================================================
if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_subdirectory(tests)
endif ()

# ==============================================================================
# Install rules
# ==============================================================================

# 1. Library archive + headers
install(TARGETS sampler
    EXPORT samplerTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 2. CMake targets file (sampler-targets.cmake)
install(EXPORT samplerTargets
    FILE        sampler-targets.cmake
    NAMESPACE   sampler::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sampler
)

# 3. Package config file (sampler-config.cmake)
configure_package_config_file(
    cmake/sampler-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sampler-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sampler
)

# 4. Version file (sampler-version.cmake)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/sampler-version.cmake
    VERSION     ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sampler-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/sampler-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sampler
)
