cmake_minimum_required(VERSION 3.14)
project(mpi_utils LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Intel|IntelLLVM)$")
    add_compile_options(-fp-model precise)
endif()

add_library(mpi_utils INTERFACE)
target_include_directories(mpi_utils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:>
)
target_compile_features(mpi_utils INTERFACE cxx_std_17)

find_package(MPI QUIET)
if(MPI_FOUND)
    target_link_libraries(mpi_utils INTERFACE MPI::MPI_CXX)
endif()
