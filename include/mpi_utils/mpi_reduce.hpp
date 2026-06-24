#ifndef MPI_UTILS_MPI_REDUCE_HPP
#define MPI_UTILS_MPI_REDUCE_HPP

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>
#include <mpi.h>
#include "types.h"
#include "MpiUtilsError.hpp"

template<typename T>
std::pair<rank_t, T> MPI_Minmax_loc(const T &data, const MPI_Comm &comm, bool max)
{
	rank_t rank;
	MPI_Comm_rank(comm, &rank);

	MPI_Datatype dtype;
	if constexpr(std::is_same_v<T, double>)
	{
		dtype = MPI_DOUBLE_INT;
	}
	else if constexpr(std::is_same_v<T, int>)
	{
		dtype = MPI_2INT;
	}
	else
	{
		MpiUtilsError eo("Unsupported type for MPI_Minmax_loc");
		eo.addEntry("Type", typeid(T).name());
		throw eo;
	}
	struct
	{
		T value;
		rank_t rank;
	} myVal = {data, rank}, outVal;
	MPI_Allreduce(&myVal, &outVal, 1, dtype, max ? MPI_MAXLOC : MPI_MINLOC, comm);
	return std::make_pair(outVal.rank, outVal.value);
}

template<typename T>
std::pair<rank_t, T> MPI_Max_loc(const T &data, const MPI_Comm &comm = MPI_COMM_WORLD)
{
	return MPI_Minmax_loc(data, comm, true);
}

template<typename T>
std::pair<rank_t, T> MPI_Min_loc(const T &data, const MPI_Comm &comm = MPI_COMM_WORLD)
{
	return MPI_Minmax_loc(data, comm, false);
}

template<typename T>
std::pair<rank_t, T> MPI_Max_loc(const std::vector<T> &data, const MPI_Comm &comm = MPI_COMM_WORLD)
{
	return MPI_Minmax_loc(data.empty()? std::numeric_limits<T>::lowest() : *std::max_element(data.begin(), data.end()), comm, true);
}

template<typename T>
std::pair<rank_t, T> MPI_Min_loc(const std::vector<T> &data, const MPI_Comm &comm = MPI_COMM_WORLD)
{
	return MPI_Minmax_loc(data.empty()? std::numeric_limits<T>::max() : *std::min_element(data.begin(), data.end()), comm, false);
}

#endif // MPI_UTILS_MPI_REDUCE_HPP
