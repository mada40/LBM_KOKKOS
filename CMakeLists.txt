cmake_minimum_required(VERSION 3.25.0)

set(CMAKE_GENERATOR_TOOLSET "Intel C++ Compiler 2023")

project(Example CXX)
add_subdirectory(kokkos)

file(GLOB samples_list RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(sample_filename ${samples_list})
  get_filename_component(sample ${sample_filename} NAME_WE)

  add_executable(${sample} ${sample_filename})
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
      target_link_libraries(${sample} OpenMP::OpenMP_CXX)
  endif()
  target_link_libraries(${sample} Kokkos::kokkos)
endforeach()



