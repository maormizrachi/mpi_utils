#ifndef BUFFERS_MANAGER_QUERY_AGENT_HPP
#define BUFFERS_MANAGER_QUERY_AGENT_HPP

#include <limits>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>
#include <mpi.h>

// set data structure:
#include <boost/container/flat_set.hpp>
#include <unordered_set>

#ifdef TIMING
    #include <chrono>
#endif // TIMING

#include "QueryAgent.hpp"
#include <mpi_utils/serialize/Serializer.hpp>
#include <mpi_utils/BuffersManager.hpp>
#include <mpi_utils/AmountManager.hpp>
#include <mpi_utils/BatchTrace.hpp>

#define TAG_REQUEST 200
#define TAG_RESPONSE 201
#define TAG_FINISHED 202

#define UNDEFINED_BUFFER_IDX -1
#define FLUSH_QUERIES_NUM 50

namespace buffers_manager_query_agent_detail
{
// Opt-in (MADVORO_FLUSH_QUERY_TAIL=1, read once per process): when a rank has submitted its last query of a batch it
// dispatches its partly filled query buffers at once instead of after the 1024-poll-cycle dispatch timeout. The
// messages, their contents and their per-peer order are unchanged (nothing is added to a buffer after the last
// query); only their timing moves. Purely local, so ranks may differ without affecting any collective.
inline bool FlushQueryTail()
{
    static const bool Enabled = []()
    {
        const char *Value = std::getenv("MADVORO_FLUSH_QUERY_TAIL");
        return Value != nullptr and std::string(Value) == "1";
    }();
    return Enabled;
}
}

template<typename QueryData, typename AnswerType>
class BuffersManagerQueryAgent : public QueryAgent<QueryData, AnswerType>
{
public:
    template<typename T>
    using _set = boost::container::flat_set<T>;

    using _subQueryData = SubQueryData<QueryData>;
    using _queryBatchInfo = QueryBatchInfo<QueryData, AnswerType>;
    using _queryInfo = QueryInfo<QueryData, AnswerType>;

    BuffersManagerQueryAgent(const TalkAgent<QueryData> *talkAgent, AnswerAgent<QueryData, AnswerType> *answerAgent, bool sendToSelf = false, const MPI_Comm &comm = MPI_COMM_WORLD);

    virtual ~BuffersManagerQueryAgent() = default;
    
    QueryBatchInfo<QueryData, AnswerType> runBatch(const std::vector<QueryData> &queries) override;
    inline std::vector<std::vector<size_t>> &getRecvData() override{return this->recvData;};
    inline std::vector<int> &getRecvProc() override{return this->recvProcessorsRanks;};
    inline void SetBatchTrace(QueryBatchTrace *Trace) override{this->BatchTrace = Trace;};

private:
   struct _answerInfo : public Serializable
   {
    public:
        size_t parent_id;
        std::vector<AnswerType> result;

        _answerInfo(size_t parent_id = std::numeric_limits<size_t>::max(), std::vector<AnswerType> result = std::vector<AnswerType>()) : parent_id(parent_id), result(result)
        {}
   
        virtual size_t load(const Serializer *serializer, size_t byteOffset) override
        {
            size_t bytes = 0;
            bytes += serializer->extract(this->parent_id, byteOffset);
            bytes += serializer->extract(this->result, byteOffset + bytes);
            return bytes;
        }

        virtual size_t dump(Serializer *serializer) const override
        {
            size_t bytes = 0;
            bytes += serializer->insert(this->parent_id);
            bytes += serializer->insert(this->result);
            return bytes;
        }
    };

    std::shared_ptr<BuffersManager<_subQueryData>> queriesBufferManager;
    std::shared_ptr<BuffersManager<_answerInfo>> answersBufferManager;
    std::shared_ptr<AmountManager> finishManager;
    std::vector<std::pair<rank_t, _subQueryData>> queriesToHandle;
    std::vector<std::pair<rank_t, _answerInfo>> answersToHandle;

    size_t receivedUntilNow; // number of answers I received until now
    size_t shouldReceiveInTotal; // number of answers I have to receive (to know when to finish)
    bool finishedMyQueries; // if finished to answer my queries
    
    // Diagnostics (SetBatchTrace): the record of the running batch, or null (off). PollCounter selects the sampled
    // progress passes and runs over the agent's life, so batches do not all start with a sampled pass.
    QueryBatchTrace *BatchTrace = nullptr;
    std::uint64_t PollCounter = 0;

    // Local progress state compared before and after a progress pass to classify it as empty or productive. No member
    // initializers: an untraced pass leaves it unset, and ReadPollProgress() assigns every member.
    struct PollProgress
    {
        size_t QueryMessagesReceived;
        size_t AnswerMessagesReceived;
        size_t QuerySendsCompleted;
        size_t AnswerSendsCompleted;
        size_t AnswersConsumed;
        AmountManager::counter_t FinishValue;
        bool FinishDone;
        bool FinishVerify;
        bool FinishPending;

        bool operator==(const PollProgress &Other) const
        {
            return QueryMessagesReceived == Other.QueryMessagesReceived and
                   AnswerMessagesReceived == Other.AnswerMessagesReceived and
                   QuerySendsCompleted == Other.QuerySendsCompleted and
                   AnswerSendsCompleted == Other.AnswerSendsCompleted and AnswersConsumed == Other.AnswersConsumed and
                   FinishValue == Other.FinishValue and FinishDone == Other.FinishDone and
                   FinishVerify == Other.FinishVerify and FinishPending == Other.FinishPending;
        }
    };

    PollProgress ReadPollProgress() const;

    std::vector<int> recvProcessorsRanks;  // a vector of ranks, that we received data from
    std::vector<std::vector<size_t>> recvData; // a vector of vectors. The vector in index `i` contains the data indices (relatively to my final answer of the batch) that sent to `this->sentProcessorsRanks[i]`.

    void receiveAnswers(QueryBatchInfo<QueryData, AnswerType> &batch);
    void answerQueries();
    void sendQuery(const QueryInfo<QueryData, AnswerType> &query);
    void rearrangeResult(_queryBatchInfo &queriesBatch);

    void sendFinish(_queryBatchInfo &queriesBatch);
};

template<typename QueryData, typename AnswerType>
BuffersManagerQueryAgent<QueryData, AnswerType>::BuffersManagerQueryAgent(const TalkAgent<QueryData> *talkAgent, AnswerAgent<QueryData, AnswerType> *answerAgent, bool sendToSelf, const MPI_Comm &comm):
        QueryAgent<QueryData, AnswerType>(talkAgent, answerAgent, sendToSelf, comm)
{
    this->finishedMyQueries = false;
}

template<typename QueryData, typename AnswerType>
typename BuffersManagerQueryAgent<QueryData, AnswerType>::PollProgress
    BuffersManagerQueryAgent<QueryData, AnswerType>::ReadPollProgress() const
{
    // Reads only: counters of the two managers, the answer counter and the termination manager's public state.
    PollProgress Progress{};
    Progress.QueryMessagesReceived = this->queriesBufferManager->GetRecvCounter();
    Progress.AnswerMessagesReceived = this->answersBufferManager->GetRecvCounter();
    const BuffersManager<_subQueryData> &Queries = *this->queriesBufferManager;
    const BuffersManager<_answerInfo> &Answers = *this->answersBufferManager;
    Progress.QuerySendsCompleted = Queries.GetSentCounter() - Queries.GetActiveSendsNumber();
    Progress.AnswerSendsCompleted = Answers.GetSentCounter() - Answers.GetActiveSendsNumber();
    Progress.AnswersConsumed = this->receivedUntilNow;
    Progress.FinishValue = this->finishManager->GetValue();
    Progress.FinishDone = this->finishManager->GetDoneRef();
    Progress.FinishVerify = this->finishManager->GetVerifyRef();
    Progress.FinishPending = this->finishManager->HasPending();
    return Progress;
}

template<typename QueryData, typename AnswerType>
void BuffersManagerQueryAgent<QueryData, AnswerType>::receiveAnswers(_queryBatchInfo &batch)
{
    if(this->receivedUntilNow >= this->shouldReceiveInTotal)
    {
        return;
    }

    std::vector<_queryInfo> &queries = batch.queriesAnswers;
    
    for(const auto &[rank, answer] : this->answersToHandle)
    {
        ++this->receivedUntilNow;
        size_t id = answer.parent_id;
        queries[id].finalResults.insert(queries[id].finalResults.end(), answer.result.begin(), answer.result.end());
        batch.dataByRanks[rank].insert(batch.dataByRanks[rank].end(), answer.result.begin(), answer.result.end());
    }
    this->answersToHandle.clear();

    if(this->finishedMyQueries and this->shouldReceiveInTotal == this->receivedUntilNow)
    {
        this->sendFinish(batch);
    }
}

template<typename QueryData, typename AnswerType>
void BuffersManagerQueryAgent<QueryData, AnswerType>::answerQueries()
{
    for(const auto &[rank, query] : this->queriesToHandle)
    {
        _answerInfo info;
        info.parent_id = query.parent_id;
        info.result = this->answerAgent->answer(query.data, rank);
        this->answersBufferManager->Add(rank, info);
    }
    this->queriesToHandle.clear();
}

template<typename QueryData, typename AnswerType>
void BuffersManagerQueryAgent<QueryData, AnswerType>::sendQuery(const _queryInfo &query)
{
    if((this->size == 1) and (!this->sendToSelf))
    {
        return;
    }
    typename TalkAgent<QueryData>::RanksSet talkingRanks = this->talkAgent->getTalkList(query.data);
    for(const int &_rank : talkingRanks)
    {
        if((_rank == this->rank) and (!this->sendToSelf))
        {
            continue; // unnecessary to send
        }
        _subQueryData subQuery;
        subQuery.data = query.data;
        subQuery.parent_id = query.id;
        this->queriesBufferManager->Add(_rank, subQuery);
        ++this->shouldReceiveInTotal;
    }
}

template<typename QueryData, typename AnswerType>
void BuffersManagerQueryAgent<QueryData, AnswerType>::sendFinish(_queryBatchInfo &queriesBatch)
{
    #ifdef TIMING
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
        queriesBatch.receivedAllTime = std::chrono::duration_cast<std::chrono::duration<double>>(now - queriesBatch.beginClockTime).count();
    #endif // TIMING

    this->finishManager->Decrease(1);
}

template<typename QueryData, typename AnswerType>
void BuffersManagerQueryAgent<QueryData, AnswerType>::rearrangeResult(_queryBatchInfo &queriesBatch)
{
     // make receive, by the order of the recv array
    for(size_t i = 0; i < this->recvProcessorsRanks.size(); i++)
    {
        int _rank = this->recvProcessorsRanks[i];
        std::vector<size_t> &rankRecvData = this->recvData[i]; // indices vector
        const std::vector<AnswerType> &newDataFromRank = queriesBatch.dataByRanks[_rank]; // the data itself
        rankRecvData.reserve(newDataFromRank.size());

        queriesBatch.result.reserve(queriesBatch.result.size() + rankRecvData.size());
        for(const AnswerType &_data : newDataFromRank)
        {
            rankRecvData.push_back(queriesBatch.result.size()); // the index of the data received by this rank
            queriesBatch.result.emplace_back(_data);
        }
    }
}

template<typename QueryData, typename AnswerType>
QueryBatchInfo<QueryData, AnswerType> BuffersManagerQueryAgent<QueryData, AnswerType>::runBatch(const std::vector<QueryData> &queries)
{
    // Diagnostics (SetBatchTrace): clock reads and local counters only, never a message or a collective. When off
    // (Trace null) each progress pass pays one test of a local pointer.
    using TraceClock = std::chrono::steady_clock;
    QueryBatchTrace *const Trace = this->BatchTrace;
    TraceClock::time_point TraceMark, SubmitEnd;
    auto TraceLap = [&TraceMark](double &Seconds)
    {
        const TraceClock::time_point Now = TraceClock::now();
        Seconds += std::chrono::duration<double>(Now - TraceMark).count();
        TraceMark = Now;
    };
    if(Trace != nullptr)
    {
        *Trace = QueryBatchTrace();
        TraceMark = TraceClock::now();
    }

    this->receivedUntilNow = 0; // reset the receive counter
    this->shouldReceiveInTotal = 0; // reset the should-be-received counter
    for(std::vector<size_t> &_receivedDataFromRank : this->recvData)
    {
        _receivedDataFromRank.clear();
    }

    auto queriesRecv = [this](const _subQueryData *newValues, size_t newValuesCount, rank_t fromRank)
    {
        for(size_t i = 0; i < newValuesCount; i++)
        {
            this->queriesToHandle.emplace_back(fromRank, newValues[i]);
        }
    };
    this->queriesBufferManager = std::make_shared<BuffersManager<_subQueryData>>(this->comm, queriesRecv, TAG_REQUEST, (sizeof(_subQueryData)) * 50, (sizeof(_subQueryData)) * 12, 1024, 1);
    auto answersRecv = [this](const _answerInfo *newValues, size_t newValuesCount, rank_t fromRank)
    {
        for(size_t i = 0; i < newValuesCount; i++)
        {
            this->answersToHandle.emplace_back(fromRank, newValues[i]);
        }
    };
    size_t sizeOfAvgAnswerInfo = sizeof(size_t) + 1024 * sizeof(AnswerType);
    this->answersBufferManager = std::make_shared<BuffersManager<_answerInfo>>(this->comm, answersRecv, TAG_RESPONSE, sizeOfAvgAnswerInfo * 64, sizeOfAvgAnswerInfo * 8, 50, 1);
    if(Trace != nullptr)
    {
        this->queriesBufferManager->SetTrace(&Trace->QueryManager);
        this->answersBufferManager->SetTrace(&Trace->AnswerManager);
    }

    size_t originalQueriesNum = queries.size();
    _queryBatchInfo queriesBatch;
    #ifdef TIMING
        queriesBatch.beginClockTime = std::chrono::system_clock::now();
    #endif // TIMING

    queriesBatch.queriesAnswers.reserve(originalQueriesNum);
    std::vector<_queryInfo> &queriesInfo = queriesBatch.queriesAnswers;
    queriesBatch.dataByRanks.resize(this->size);
    this->finishedMyQueries = queries.empty();
    size_t i = 0;

    if(Trace != nullptr)
    {
        TraceLap(Trace->ConstructSeconds);
    }
    this->finishManager = std::make_shared<AmountManager>(this->comm);
    if(Trace != nullptr)
    {
        TraceLap(Trace->ConstructorBarrierSeconds);
        SubmitEnd = TraceMark; // stays the loop start when this rank has no query to submit
    }

    // Every rank starts with exactly one pending finish, so the root's total is `size`: no reduction needed.
    this->finishManager->InitializeUniform(1);

    const bool &finished = this->finishManager->GetDoneRef();
    const bool &verify = this->finishManager->GetVerifyRef();
    
    // if doesn't have any queries, send a finish message
    if(this->finishedMyQueries)
    {
        this->sendFinish(queriesBatch);
        #ifdef TIMING
            std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
            queriesBatch.finishSubmittingTime = std::chrono::duration_cast<std::chrono::duration<double>>(now - queriesBatch.beginClockTime).count();
        #endif // TIMING
    }

    // size_t lastReceived = 0;
    while((!this->finishedMyQueries) or not finished)
    {
        bool InSubmit = false;
        bool Sampled = false;
        size_t ToAnswer = 0;
        PollProgress Before;
        TraceClock::time_point PollStart, PollQueryEnd, ServeStart;
        if(Trace != nullptr)
        {
            InSubmit = !this->finishedMyQueries;
            ToAnswer = this->queriesToHandle.size(); // exactly what answerQueries() below answers
            Before = this->ReadPollProgress();
            Sampled = ((this->PollCounter++ % QueryBatchTracePollSamplePeriod) == 0);
            if(Sampled)
            {
                PollStart = TraceClock::now();
            }
        }

        if(!this->finishedMyQueries)
        {
            const QueryData &queryData = queries[i];
            queriesInfo.push_back({queryData, queriesInfo.size(), std::vector<AnswerType>()});
            _queryInfo &query = queriesInfo.back();

            this->sendQuery(query);
            if(queriesInfo.size() == originalQueriesNum)
            {
                #ifdef TIMING
                    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
                    queriesBatch.finishSubmittingTime = std::chrono::duration_cast<std::chrono::duration<double>>(now - queriesBatch.beginClockTime).count();
                #endif // TIMING

                this->finishedMyQueries = true;
                if(buffers_manager_query_agent_detail::FlushQueryTail())
                {
                    this->queriesBufferManager->DispatchAll(); // no further Add to these buffers in this batch
                }
                // if had several queries, but no communication was needed, send a finish message
                if(this->shouldReceiveInTotal == this->receivedUntilNow)
                {
                    this->sendFinish(queriesBatch);
                }
            }
        }
        
        // if(this->rank == 0)
        // {
        //     std::cout << "Finished value is " << this->finishManager->GetValue() << std::endl; 
        // }
        if(Sampled and InSubmit)
        {
            PollQueryEnd = TraceClock::now();
        }
        this->receiveAnswers(queriesBatch);
        if(ToAnswer > 0)
        {
            ServeStart = TraceClock::now();
        }
        this->answerQueries();
        if(ToAnswer > 0)
        {
            Trace->ServeSeconds += std::chrono::duration<double>(TraceClock::now() - ServeStart).count();
            Trace->QueriesAnswered += ToAnswer;
        }
        this->answersBufferManager->HandleIncomingOutcoming();
        this->queriesBufferManager->HandleIncomingOutcoming();
        this->finishManager->Progress();
        if(verify)
        {
            bool ok = this->finishedMyQueries and this->queriesBufferManager->CountOutcoming() == 0 and this->answersBufferManager->CountOutcoming() == 0;
            this->finishManager->Verify(ok);
        }

        if(Trace != nullptr)
        {
            const bool Productive = (ToAnswer > 0) or !(this->ReadPollProgress() == Before);
            const size_t Class = Productive ? 1 : 0;
            TraceClock::time_point PollEnd;
            if(Sampled or (InSubmit and this->finishedMyQueries))
            {
                PollEnd = TraceClock::now();
            }
            if(InSubmit)
            {
                ++Trace->SubmitPolls;
                Trace->SubmitPollsEmpty += Productive ? 0 : 1;
                if(Sampled)
                {
                    ++Trace->SubmitSampled[Class];
                    Trace->SubmitSampledSeconds[Class] += std::chrono::duration<double>(PollEnd - PollStart).count();
                    Trace->SubmitSampledQuerySeconds += std::chrono::duration<double>(PollQueryEnd - PollStart).count();
                }
                if(this->finishedMyQueries)
                {
                    SubmitEnd = PollEnd;
                }
            }
            else
            {
                ++Trace->DrainPolls;
                Trace->DrainPollsEmpty += Productive ? 0 : 1;
                if(Sampled)
                {
                    ++Trace->DrainSampled[Class];
                    Trace->DrainSampledSeconds[Class] += std::chrono::duration<double>(PollEnd - PollStart).count();
                }
            }
        }

        // if(lastReceived != this->receivedUntilNow)
        // {
        //     lastReceived = this->receivedUntilNow;
        //     std::cout << "Rank " << this->rank << " received " << lastReceived << " answers until now, should receive " << this->shouldReceiveInTotal << std::endl;
        // }
        ++i;
    }

    #ifdef TIMING
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
        queriesBatch.finishTime = std::chrono::duration_cast<std::chrono::duration<double>>(now - queriesBatch.beginClockTime).count();
    #endif // TIMING

    if(Trace != nullptr)
    {
        const TraceClock::time_point LoopStart = TraceMark;
        TraceMark = TraceClock::now();
        Trace->SubmitSeconds = std::chrono::duration<double>(SubmitEnd - LoopStart).count();
        Trace->DrainSeconds = std::chrono::duration<double>(TraceMark - SubmitEnd).count();
    }

    // add to the list the processors that sent us a message for the first time
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        if(_rank == this->rank)
        {
            continue; // shouldn't be relevant
        }
        if(queriesBatch.dataByRanks[_rank].empty())
        {
            continue; // the rank `_rank` did not send us any message
        }
        size_t rankIndex = std::find(this->recvProcessorsRanks.begin(), this->recvProcessorsRanks.end(), _rank) - this->recvProcessorsRanks.begin();
        if(rankIndex == this->recvProcessorsRanks.size())
        {
            // rank is not inside the recvProcessors rank, add it
            this->recvProcessorsRanks.push_back(_rank);
            this->recvData.emplace_back(std::vector<size_t>());
        }
    }

    // Formerly queries->Destroy(), answers->Destroy(), then this barrier: two reduce-scatters and three barriers. Now one
    // reduce-scatter carries both managers' counters and the barrier below is the only one. At termination every
    // message has been received (each query is answered and each answer received before the finish count reaches
    // zero), so the drains inside never run; the barrier still follows every rank's receive cancellation (and the
    // AmountManager's, via its destructor), as before.
    if(Trace != nullptr)
    {
        TraceLap(Trace->RearrangeSeconds);
    }
    DestroyBuffersManagerPair(*this->queriesBufferManager, *this->answersBufferManager, false);
    this->finishManager = nullptr;
    if(Trace != nullptr)
    {
        TraceLap(Trace->TeardownSeconds);
    }

    MPI_Barrier(this->comm);
    if(Trace != nullptr)
    {
        TraceLap(Trace->FinalBarrierSeconds);
    }

    this->rearrangeResult(queriesBatch);

    if(Trace != nullptr)
    {
        TraceLap(Trace->RearrangeSeconds);
        Trace->SubQueriesSent = this->shouldReceiveInTotal;
        Trace->AnswersReceived = this->receivedUntilNow;
        this->queriesBufferManager->SetTrace(nullptr);
        this->answersBufferManager->SetTrace(nullptr);
    }

    return queriesBatch;
}


#endif // BUFFERS_MANAGER_QUERY_AGENT_HPP
