add_sampler_test(test_so_isotropic_sampler)
add_sampler_test(test_so_anisotropic_sampler)
add_sampler_test(test_phase2_acceptance_rate)
add_sampler_test(test_so_anisotropic_heterogeneous_sampler)

# ==============================================================================
# HDF5-dependent visualisation test
# ==============================================================================
if(APPLE AND NOT DEFINED HDF5_ROOT)
    set(HDF5_ROOT /opt/homebrew/opt/hdf5)
endif()

find_package(HDF5 COMPONENTS C QUIET)

if(HDF5_FOUND)
    add_executable(test_phase2_visualise
        ${CMAKE_CURRENT_SOURCE_DIR}/test_phase2_visualise.cpp)
    target_link_libraries(test_phase2_visualise PRIVATE sampler HDF5::HDF5)
    target_compile_definitions(test_phase2_visualise PRIVATE
        SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    message(STATUS "HDF5 found — building test_phase2_visualise")
else()
    message(WARNING "HDF5 not found — skipping test_phase2_visualise")
endif()
