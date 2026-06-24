#ifndef MPI_UTILS_EXCHANGE_HPP
#define MPI_UTILS_EXCHANGE_HPP

#include <mpi.h>
#include <functional>
#include "serialize/Serializer.hpp"
#include "mpi_commands.hpp"

template<typename T>
struct ExchangeAnswer
{
    std::vector<T> output;
    std::vector<size_t> indicesToMe;
    std::vector<int> processesSend;
    std::vector<std::vector<size_t>> indicesToProcesses;
    std::vector<int> processesRecv;
    std::vector<std::vector<T>> answerByProcesses;
};

template<typename T, typename ExchangeDetermineFunc = std::function<int(const T&)>>
ExchangeAnswer<T> dataExchange(const std::vector<T> &data, const ExchangeDetermineFunc &getOwner, const MPI_Comm &comm = MPI_COMM_WORLD)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<MPI_Request> requests;
    ExchangeAnswer<T> answer{};

    std::vector<std::vector<T>> dataByProcesses(size);
    std::vector<std::vector<size_t>> indicesToAllProcesses(size);

    size_t N = data.size();
    for(size_t i = 0; i < N; i++)
    {
        rank_t _rank = getOwner(data[i]);
        if(_rank != rank)
        {
            dataByProcesses[_rank].push_back(data[i]);
            indicesToAllProcesses[_rank].push_back(i);
        }
        else
        {
            answer.indicesToMe.push_back(i);
            answer.output.push_back(data[i]);
        }
    }

    for(rank_t _rank = 0; _rank < size; _rank++)
    {
        if(_rank == rank or dataByProcesses[_rank].empty())
        {
            continue;
        }
        answer.processesSend.push_back(_rank);
        answer.indicesToProcesses.emplace_back(std::move(indicesToAllProcesses[_rank]));
    }
    
    
    std::vector<std::vector<T>> incomingData = MPI_Exchange_all_to_all(dataByProcesses, comm);
    
    for(rank_t _rank = 0; _rank < size; _rank++)
    {
        if(_rank == rank or incomingData[_rank].empty())
        {
            continue;
        }

        if(std::find(answer.processesSend.begin(), answer.processesSend.end(), _rank) == answer.processesSend.end())
        {
            answer.processesSend.push_back(_rank);
            answer.indicesToProcesses.emplace_back(std::vector<size_t>());
        }

        answer.processesRecv.push_back(_rank);
        answer.answerByProcesses.emplace_back(std::vector<T>());
    }

    size_t answerSize = answer.processesSend.size();
    for(size_t i = 0; i < answerSize; i++)
    {
        rank_t _rank = answer.processesSend[i];
        size_t idx = std::distance(answer.processesRecv.cbegin(), std::find(answer.processesRecv.cbegin(), answer.processesRecv.cend(), _rank));
        if(idx == answer.processesRecv.size())
        {
            continue;
        }
        const std::vector<T> &incomingDataOfRank = incomingData[_rank];
        answer.answerByProcesses[idx] = std::move(incomingDataOfRank);
        answer.output.insert(answer.output.end(), incomingDataOfRank.cbegin(), incomingDataOfRank.cend());
    }
    return answer;
}

#endif // MPI_UTILS_EXCHANGE_HPP
