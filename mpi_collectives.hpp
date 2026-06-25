#ifndef MPI_UTILS_MPI_COLLECTIVES_HPP
#define MPI_UTILS_MPI_COLLECTIVES_HPP

#include <cassert>
#include <vector>
#include <mpi.h>
#include "serialize/Serializer.hpp"

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<std::vector<T>> MPI_All_cast_by_ranks(const Container<T, Ts...> &data, const MPI_Comm &comm = MPI_COMM_WORLD)
{
    rank_t size;
    MPI_Comm_size(comm, &size);

    Serializer send;
    send.reset();
    int count = static_cast<int>(send.insert_all(data));
    std::vector<int> recvCounts(size, 0);
    MPI_Allgather(&count, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm);

    std::vector<int> recvDisplacements(size, 0);
    size_t totalToReceive = 0;
    for(rank_t _rank = 0; _rank < size; _rank++)
    {
        totalToReceive += static_cast<size_t>(recvCounts[_rank]);
        if(_rank > 0)
        {
            recvDisplacements[_rank] = recvDisplacements[_rank - 1] + recvCounts[_rank - 1];
        }
    }

    std::vector<int> sendDisplacements(size, 0);
    std::vector<int> sendCounts(size, count);
    Serializer recv;
    recv.reset();
    recv.resize(totalToReceive);
    MPI_Alltoallv(send.getData(), sendCounts.data(), sendDisplacements.data(), MPI_BYTE,
                    recv.getData(), recvCounts.data(), recvDisplacements.data(), MPI_BYTE, comm);

    std::vector<std::vector<T>> resultByRanks(size);
    for(rank_t _rank = 0; _rank < size; _rank++)
    {
        size_t readCount = recv.extract(resultByRanks[_rank], recvDisplacements[_rank], recvCounts[_rank]);
        assert(readCount == recvCounts[_rank]);
    }
    return resultByRanks;
}

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<T> MPI_All_cast(const Container<T, Ts...> &data, const MPI_Comm &comm)
{
	std::vector<std::vector<T>> resultByRanks = MPI_All_cast_by_ranks(data, comm);
	std::vector<T> result;
	for(const std::vector<T> &values : resultByRanks)
	{
		result.insert(result.end(), values.cbegin(), values.cend());
	}
	return result;
}

template<typename T>
T MPI_Bcast_serializable(const T &data, rank_t owner, const MPI_Comm &comm = MPI_COMM_WORLD)
{
    rank_t rank;
    MPI_Comm_rank(comm, &rank);

    Serializer buf;
    buf.reset();
    size_t sizeSent = 0;
    if(rank == owner)
    {
        sizeSent = buf.insert(data);
    }
    MPI_Bcast(&sizeSent, 1, MPI_UNSIGNED_LONG_LONG, owner, comm);

    if(rank != owner)
    {
        buf.resize(sizeSent);
    }
    MPI_Bcast(buf.getData(), sizeSent, MPI_BYTE, owner, comm);

    T value;
    size_t sizeRead = buf.extract(value, 0);
    assert(sizeRead == sizeSent);
    return value;
}

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<T> MPI_Gatherv_serializable(const Container<T, Ts...> &data, rank_t root, const MPI_Comm &comm)
{
    rank_t rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(size == 1)
    {
        return data;
    }
    Serializer send;
    send.reset();
    int bytes = static_cast<int>(send.insert_all(data));

    if(rank == root)
    {
        std::vector<int> toRecvBytes(size);
        MPI_Gather(&bytes, 1, MPI_INT, toRecvBytes.data(), 1, MPI_INT, root, comm);
        std::vector<int> toRecvDisplacements(size, 0);
        size_t totalSize = 0;
        for(int _rank = 0; _rank < size; _rank++)
        {
            totalSize += toRecvBytes[_rank];
            if(_rank > 0)
            {
                toRecvDisplacements[_rank] = toRecvDisplacements[_rank - 1] + toRecvBytes[_rank - 1];
            }
        }
        Serializer recv;
        recv.reset();
        recv.resize(totalSize);
        MPI_Gatherv(send.getData(), bytes, MPI_BYTE, recv.getData(), toRecvBytes.data(), toRecvDisplacements.data(), MPI_BYTE, root, comm);
        std::vector<T> toReturn;
        recv.extract_all(toReturn);
        return toReturn;
    }
    MPI_Gather(&bytes, 1, MPI_INT, NULL, 0, MPI_INT, root, comm);
    MPI_Gatherv(send.getData(), bytes, MPI_BYTE, NULL, NULL, NULL, MPI_BYTE, root, comm);
    return std::vector<T>();
}

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<T> MPI_Spread(const Container<T, Ts...> &data, rank_t root, const MPI_Comm &comm)
{
	rank_t rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	if(size == 1)
	{
		return data;
	}

	Serializer send;
	Serializer recv;
	send.reset();
	recv.reset();
	int mySize;
	if(rank == root)
	{
		size_t totalSize = data.size();
		size_t idealSize = totalSize / size;
		std::vector<int> counts(size, 0);
		std::vector<int> offsets(size, 0);
		size_t current = 0;
		for(rank_t _rank = 0; _rank < size; _rank++)
		{
			size_t _begin = _rank * idealSize;
			size_t _end = (_rank == size - 1)? totalSize : ((_rank + 1) * idealSize);
			size_t length = _end - _begin;
			offsets[_rank] = current;
			counts[_rank] = send.insert_elements(data, _begin, length);
            current += counts[_rank];
		}
		MPI_Scatter(counts.data(), 1, MPI_INT, &mySize, 1, MPI_INT, root, comm);
		recv.resize(mySize);
		MPI_Scatterv(send.getData(), counts.data(), offsets.data(), MPI_BYTE, recv.getData(), mySize, MPI_BYTE, root, comm);
	}
	else
	{
		MPI_Scatter(NULL, 1, MPI_INT, &mySize, 1, MPI_INT, root, comm);
		recv.resize(mySize);
		MPI_Scatterv(NULL, NULL, NULL, MPI_BYTE, recv.getData(), mySize, MPI_BYTE, root, comm);
	}

	std::vector<T> toReturn;
	recv.extract_all(toReturn);
	return toReturn;
}

#endif // MPI_UTILS_MPI_COLLECTIVES_HPP
