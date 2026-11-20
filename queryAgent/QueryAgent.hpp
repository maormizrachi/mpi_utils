#ifndef QUERY_AGENT_HPP
#define QUERY_AGENT_HPP


#include <mpi.h>

#include "AnswerAgent.hpp"
#include "TalkAgent.hpp"
#include "QueryData.hpp"
#include <mpi_utils/BatchTrace.hpp>

template<typename QueryData, typename AnswerType>
class QueryAgent
{
public:
    QueryAgent(const TalkAgent<QueryData> *talkAgent, AnswerAgent<QueryData, AnswerType> *answerAgent, bool sendToSelf = false, const MPI_Comm &comm = MPI_COMM_WORLD);
    
    virtual ~QueryAgent() = default;

    virtual QueryBatchInfo<QueryData, AnswerType> runBatch(const std::vector<QueryData> &queries) = 0;
    
    virtual std::vector<std::vector<size_t>> &getRecvData() = 0;
    
    virtual std::vector<int> &getRecvProc() = 0;

    // Diagnostics only: while Trace is non-null, runBatch writes its per-rank counters and timers into it (reset at the
    // start of every batch). Agents without instrumentation ignore it.
    virtual void SetBatchTrace(QueryBatchTrace *Trace) {(void)Trace;}

protected:
    MPI_Comm comm;
    int rank, size;
    const TalkAgent<QueryData> *talkAgent; // answers to whom we should talk to, given a query
    AnswerAgent<QueryData, AnswerType> *answerAgent; // an answer agent
    bool sendToSelf; // should send queries to self
};

template<typename QueryData, typename AnswerType>
QueryAgent<QueryData, AnswerType>::QueryAgent(const TalkAgent<QueryData> *talkAgent, AnswerAgent<QueryData, AnswerType> *answerAgent, bool sendToSelf, const MPI_Comm &comm)
{
    MPI_Comm_size(comm, &this->size);
    MPI_Comm_rank(comm, &this->rank);
    this->talkAgent = talkAgent;
    this->answerAgent = answerAgent;
    this->sendToSelf = sendToSelf;
    this->comm = comm;
}


#endif // QUERY_AGENT_HPP