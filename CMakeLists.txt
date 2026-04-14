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
else ()
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

# sampler:: namespace alias so consumers can write target_link_libraries(... sampler::sampler)
add_library(sampler::sampler ALIAS sampler)

target_include_directories(sampler
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(sampler PUBLIC isomorphism OpenMP::OpenMP_CXX)

# ==============================================================================
# Build-type flags
# ==============================================================================
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(sampler PRIVATE -O3 -ffast-math -march=native)
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=undefined)
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
