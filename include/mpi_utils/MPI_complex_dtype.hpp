#ifndef MPI_UTILS_MPI_COMPLEX_DTYPE_HPP
#define MPI_UTILS_MPI_COMPLEX_DTYPE_HPP

#include <mpi.h>
#include <type_traits>

template<typename T>
struct MPI_has_complex_dtype : std::false_type {};

#endif // MPI_UTILS_MPI_COMPLEX_DTYPE_HPP
