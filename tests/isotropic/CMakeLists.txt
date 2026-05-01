add_sampler_test(test_stiefel_sampler)
add_sampler_test(test_so_sampler)

# ==============================================================================
# HDF5-dependent tests
# ==============================================================================
if(APPLE AND NOT DEFINED HDF5_ROOT)
    set(HDF5_ROOT /opt/homebrew/opt/hdf5)
endif()

find_package(HDF5 COMPONENTS C QUIET)

if(HDF5_FOUND)
    add_executable(test_so_angle_sampler_save
        ${CMAKE_CURRENT_SOURCE_DIR}/test_so_angle_sampler_save.cpp)
    target_link_libraries(test_so_angle_sampler_save PRIVATE sampler HDF5::HDF5)
    message(STATUS "HDF5 found — building test_so_angle_sampler_save")
else()
    message(WARNING "HDF5 not found — skipping test_so_angle_sampler_save")
endif()
