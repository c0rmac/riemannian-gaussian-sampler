function(add_sampler_test name)
    add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)
    target_link_libraries(${name} PRIVATE sampler)
endfunction()

add_subdirectory(isotropic)
add_subdirectory(anisotropic)
