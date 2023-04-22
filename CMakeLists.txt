# Kokkos minimally requires 3.25 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.25)

# Project can mix languages - must have C++ support
# Kokkos flags are only applied to C++ files
project(Kokkos_real CXX)
# We build kokkos as a subdirectory of our project
add_subdirectory(include/kokkos)
file(GLOB samples_list RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(sample_filename ${samples_list})
  # Get file name without extension
  get_filename_component(sample ${sample_filename} NAME_WE)

  # Add and configure executable file to be produced
  add_executable(${sample} ${sample_filename})
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
      target_link_libraries(${sample} OpenMP::OpenMP_CXX)
  endif()
  target_link_libraries(${sample} Kokkos::kokkos)
endforeach()

