#ifndef MPI_UTILS_BUFFERS_MANAGER_HPP
#define MPI_UTILS_BUFFERS_MANAGER_HPP

#ifdef __WITH_MPI

#include <cassert>
#include <limits>
#include <stdexcept>
#include <mpi.h>
#include <vector>
#include <functional>
#include <boost/container/flat_map.hpp>
#include "types.h"
#include "serialize/Serializer.hpp"
#include "BatchTrace.hpp"

template<typename T>
class BuffersManager
{
public:
    BuffersManager(const MPI_Comm &comm, const std::function<void(const T *newValues, size_t newValuesCount, rank_t fromRank)> &receiveCallback, int tag, size_t buffersSize, size_t minSizeToDispatch, size_t minCyclesToDispatch, size_t initialReceiveBuffers, const std::vector<rank_t> &neighbors = std::vector<rank_t>());

    ~BuffersManager();

    void Add(rank_t rank, const T &value);

    void Receive(bool ignore = false);

    void HandleIncomingOutcoming();

    void Destroy(void);

    // Destroy() in two local steps around its one collective, so several managers on the same communicator can share
    // one counter reduction and one barrier (DestroyBuffersManagerPair). CompleteSends marks the manager destroyed and
    // completes its sends; CloseReceives takes the number of messages sent to this rank (the reduce-scatter result).
    void CompleteSends(void);

    void CloseReceives(size_t sentToMe);

    // Dispatches every non-empty pending send buffer now, in ascending rank order. Each buffer goes out as the same
    // message a later ShouldSend() would have sent, provided nothing more is added to it in between.
    void DispatchAll(void);

    inline MPI_Comm GetComm() const{return this->comm;};

    inline bool IsDestroyed() const{return this->destroyed;};

    inline size_t CountOutcoming() const{return this->sendRequests.size();};

    inline size_t GetPendingNumber() const{return this->ranksSendBuffers.size();};

    inline size_t GetActiveSendsNumber() const{return this->activeSendRequests;};

    inline size_t GetSentCounter() const {return this->sendCounter;};

    inline size_t GetRecvCounter() const {return this->recvCounter;};

    inline const std::vector<size_t> &GetAllSendCounters() const {return this->sendCounters;};

    inline const std::vector<size_t> &GetAllRecvCounters() const {return this->recvCounters;};

    inline size_t GetTotalMemoryBytes() const {return this->buffers.size() * this->buffersSize;};

    // Diagnostics only: while Trace is non-null, dispatches (by reason), messages and bytes are counted into it.
    inline void SetTrace(BuffersManagerTrace *Trace){this->TraceSink = Trace;};

private:
    MPI_Comm comm;
    bool destroyed;
    BuffersManagerTrace *TraceSink = nullptr;
    rank_t rank_world, size_world;
    std::vector<Serializer> buffers;
    std::vector<MPI_Request> sendRequests;
    size_t activeSendRequests;
    std::vector<MPI_Request> receiveRequests;
    std::vector<rank_t> receiveSources;
    boost::container::flat_set<size_t> availableBuffersIndices;
    boost::container::flat_map<size_t, size_t> sendBuffersByRequests;
    boost::container::flat_map<size_t, size_t> recvBuffersByRequests;
    std::vector<size_t> buffersToCycles;

    boost::container::flat_map<rank_t, size_t> ranksSendBuffers;

    size_t sendCounter;
    size_t recvCounter;

    std::vector<size_t> sendCounters;
    std::vector<size_t> recvCounters;

    void Dispatch(rank_t rank);

    bool ShouldSend(rank_t rank);

    void CleanSendRequests();

    size_t buffersSize;
    size_t minSizeToDispatch;
    size_t minCyclesToDispatch;
    int tag;
    std::function<void(const T *newValues, size_t newValuesCount, rank_t fromRank)> receiveCallback;
};

template<typename T>
BuffersManager<T>::BuffersManager(const MPI_Comm &comm, const std::function<void(const T *newValues, size_t newValuesCount, rank_t fromRank)> &receiveCallback, int tag, size_t buffersSize, size_t minSizeToDispatch, size_t minCyclesToDispatch, size_t numReceiveBuffers, const std::vector<rank_t> &neighbors)
    : comm(comm), destroyed(false), receiveCallback(receiveCallback), tag(tag), buffersSize(buffersSize), minSizeToDispatch(minSizeToDispatch), minCyclesToDispatch(minCyclesToDispatch)
{
    this->buffersSize += sizeof(size_t);
    this->minSizeToDispatch += sizeof(size_t);

    MPI_Comm_rank(comm, &this->rank_world);
    MPI_Comm_size(comm, &this->size_world);

    this->activeSendRequests = 0;
    this->sendCounter = 0;
    this->recvCounter = 0;

    this->sendCounters.resize(this->size_world, 0);
    this->recvCounters.resize(this->size_world, 0);

    if(this->buffersSize < this->minSizeToDispatch)
    {
        throw std::runtime_error("BuffersManager: buffersSize (" + std::to_string(this->buffersSize - sizeof(size_t))
                                 + ") is less than minSizeToDispatch (" + std::to_string(this->minSizeToDispatch - sizeof(size_t)) + ")");
    }

    if(neighbors.empty() and numReceiveBuffers == 0)
    {
        throw std::runtime_error("BuffersManager: numReceiveBuffers is 0");
    }

    if(neighbors.empty())
    {
        this->receiveSources.assign(numReceiveBuffers, MPI_ANY_SOURCE);
    }
    else
    {
        this->receiveSources = neighbors;
    }
    this->receiveRequests.resize(this->receiveSources.size(), MPI_REQUEST_NULL);
    for(size_t i = 0; i < this->receiveRequests.size(); i++)
    {
        Serializer &serializer = this->buffers.emplace_back();
        serializer.resize(this->buffersSize);
        MPI_Request &request = this->receiveRequests[i];
        MPI_Irecv(serializer.getData(), serializer.size(), MPI_BYTE, this->receiveSources[i], this->tag, comm, &request);
        assert(request != MPI_REQUEST_NULL);
        this->recvBuffersByRequests[i] = i;
    }

    this->buffersToCycles = std::vector<size_t>(this->size_world, 0);
}

template<typename T>
BuffersManager<T>::~BuffersManager()
{
    this->Destroy();
}

template<typename T>
void BuffersManager<T>::Destroy(void)
{
    if(this->destroyed)
    {
        return;
    }
    this->CompleteSends();

    size_t sentToMe = 0;
    std::vector<int> counts(this->size_world, 1);
    MPI_Reduce_scatter(this->sendCounters.data(), &sentToMe, counts.data(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, this->comm);

    this->CloseReceives(sentToMe);
    MPI_Barrier(this->comm);
}

template<typename T>
void BuffersManager<T>::CompleteSends(void)
{
    this->destroyed = true;
    if(not this->sendRequests.empty())
    {
        MPI_Waitall(this->sendRequests.size(), this->sendRequests.data(), MPI_STATUSES_IGNORE);
        this->activeSendRequests = 0;
    }

    if(this->activeSendRequests != 0)
    {
        throw std::runtime_error("BuffersManager: not all send requests were handled (active=" + std::to_string(this->activeSendRequests) + ")");
    }
}

template<typename T>
void BuffersManager<T>::CloseReceives(size_t sentToMe)
{
    if(sentToMe < this->recvCounter)
    {
        throw std::runtime_error("BuffersManager: too many receives handled (recv=" + std::to_string(this->recvCounter) + ", sentToMe=" + std::to_string(sentToMe) + ")");
    }
    else if(sentToMe > this->recvCounter)
    {
        std::vector<int> array_of_indices(this->receiveRequests.size());
        std::vector<MPI_Status> array_of_statuses(this->receiveRequests.size());
        while(this->recvCounter < sentToMe)
        {
            int outcount;
            MPI_Waitsome(this->receiveRequests.size(), this->receiveRequests.data(), &outcount, array_of_indices.data(), array_of_statuses.data());
            if(outcount == MPI_UNDEFINED)
            {
                throw std::runtime_error("BuffersManager: no active receive requests while draining (recv=" + std::to_string(this->recvCounter) + ", sentToMe=" + std::to_string(sentToMe) + ")");
            }
            for(int i = 0; i < outcount; i++)
            {
                size_t requestIndex = array_of_indices[i];
                rank_t fromRank = array_of_statuses[i].MPI_SOURCE;
                size_t bufferIndex = this->recvBuffersByRequests.at(requestIndex);
                Serializer &serializer = this->buffers[bufferIndex];
                MPI_Request &request = this->receiveRequests[requestIndex];
                this->recvCounter++;
                this->recvCounters[fromRank]++;
                if(this->TraceSink != nullptr)
                {
                    int ReceivedBytes = 0;
                    MPI_Get_count(&array_of_statuses[i], MPI_BYTE, &ReceivedBytes); // local status query
                    ++this->TraceSink->MessagesReceived;
                    this->TraceSink->BytesReceived += static_cast<size_t>(ReceivedBytes);
                }
                if(this->recvCounter < sentToMe)
                {
                    MPI_Irecv(serializer.getData(), serializer.size(), MPI_BYTE, this->receiveSources[requestIndex], this->tag, this->comm, &request);
                }
            }
        }
    }

    if(this->recvCounter != sentToMe)
    {
        throw std::runtime_error("BuffersManager: not all receive requests handled (recv=" + std::to_string(this->recvCounter) + ", sentToMe=" + std::to_string(sentToMe) + ")");
    }
    for(MPI_Request &request : this->receiveRequests)
    {
        if(request != MPI_REQUEST_NULL)
        {
            MPI_Cancel(&request);
        }
    }
    if(not this->receiveRequests.empty())
    {
        MPI_Waitall(this->receiveRequests.size(), this->receiveRequests.data(), MPI_STATUSES_IGNORE);
    }
}

template<typename T>
void BuffersManager<T>::DispatchAll(void)
{
    std::vector<rank_t> PendingRanks;
    PendingRanks.reserve(this->ranksSendBuffers.size());
    for(const auto &Pending : this->ranksSendBuffers)
    {
        size_t BufferBytes = 0;
        this->buffers[Pending.second].extract(BufferBytes, 0);
        if(BufferBytes != sizeof(size_t))
        {
            PendingRanks.push_back(Pending.first); // the same emptiness test as ShouldSend()
        }
    }
    for(rank_t PendingRank : PendingRanks)
    {
        if(this->TraceSink != nullptr)
        {
            ++this->TraceSink->DispatchByFlush;
        }
        this->Dispatch(PendingRank);
    }
}

/**
 * Destroys two managers on the same communicator with one counter reduce-scatter instead of two and without the two
 * per-manager barriers: the same local steps as First.Destroy() followed by Second.Destroy(), except that both sends
 * are completed before the shared reduce-scatter. Each rank's two received-message totals are the same numbers the two
 * separate reductions give. When FinalBarrier is false the caller must issue a barrier on the communicator before any
 * rank reuses either tag (e.g. a following MPI_Barrier), which restores the guarantee Destroy()'s barrier gave.
 */
template<typename T1, typename T2>
void DestroyBuffersManagerPair(BuffersManager<T1> &First, BuffersManager<T2> &Second, bool FinalBarrier = true)
{
    static_assert(sizeof(size_t) == sizeof(unsigned long long), "send counters are reduced as MPI_UNSIGNED_LONG_LONG");
    if(First.IsDestroyed() or Second.IsDestroyed())
    {
        throw std::runtime_error("DestroyBuffersManagerPair: a manager was already destroyed");
    }
    const MPI_Comm Comm = First.GetComm();
    int Size = 0;
    MPI_Comm_size(Comm, &Size);

    First.CompleteSends();
    Second.CompleteSends();

    const std::vector<size_t> &FirstCounters = First.GetAllSendCounters();
    const std::vector<size_t> &SecondCounters = Second.GetAllSendCounters();
    std::vector<size_t> InterleavedCounters(2 * static_cast<size_t>(Size));
    for(int Rank = 0; Rank < Size; Rank++)
    {
        InterleavedCounters[2 * Rank] = FirstCounters[Rank];
        InterleavedCounters[2 * Rank + 1] = SecondCounters[Rank];
    }
    size_t SentToMe[2] = {0, 0};
    std::vector<int> RecvCounts(Size, 2);
    MPI_Reduce_scatter(InterleavedCounters.data(), SentToMe, RecvCounts.data(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, Comm);

    First.CloseReceives(SentToMe[0]);
    Second.CloseReceives(SentToMe[1]);
    if(FinalBarrier)
    {
        MPI_Barrier(Comm);
    }
}

template<typename T>
void BuffersManager<T>::Receive(bool ignore)
{
    static std::vector<int> array_of_indices;
    static std::vector<MPI_Status> array_of_statuses;
    static std::vector<T> data;

    if(array_of_indices.size() != this->receiveRequests.size())
    {
        array_of_indices.resize(this->receiveRequests.size());
    }
    if(array_of_statuses.size() != this->receiveRequests.size())
    {
        array_of_statuses.resize(this->receiveRequests.size());
    }

    int outcount;
    MPI_Testsome(this->receiveRequests.size(), this->receiveRequests.data(), &outcount, array_of_indices.data(), array_of_statuses.data());

    for(int i = 0; i < outcount; i++)
    {
        size_t requestIndex = array_of_indices[i];
        MPI_Status &status = array_of_statuses[i];
        rank_t fromRank = status.MPI_SOURCE;
        size_t bufferIndex = this->recvBuffersByRequests.at(requestIndex);
        Serializer &serializer = this->buffers[bufferIndex];

        MPI_Request &request = this->receiveRequests[requestIndex];
        if(not ignore)
        {
            size_t n;
            size_t bytes = serializer.extract(n, 0);
            data.clear();
            serializer.extract(data, bytes, n);
            this->receiveCallback(data.data(), data.size(), fromRank);
        }

        this->recvBuffersByRequests[requestIndex] = bufferIndex;
        this->recvCounter++;
        this->recvCounters[fromRank]++;
        if(this->TraceSink != nullptr)
        {
            int ReceivedBytes = 0;
            MPI_Get_count(&status, MPI_BYTE, &ReceivedBytes); // local status query
            ++this->TraceSink->MessagesReceived;
            this->TraceSink->BytesReceived += static_cast<size_t>(ReceivedBytes);
        }
        MPI_Irecv(serializer.getData(), serializer.size(), MPI_BYTE, this->receiveSources[requestIndex], this->tag, this->comm, &request);
    }
}

template<typename T>
void BuffersManager<T>::Add(rank_t rank, const T &value)
{
    auto it = this->ranksSendBuffers.find(rank);
    size_t bufferIndex;
    if(it == this->ranksSendBuffers.end())
    {
        if(this->availableBuffersIndices.empty())
        {
            bufferIndex = this->buffers.size();
            Serializer &serializer = this->buffers.emplace_back();
            serializer.insert(static_cast<size_t>(0));
        }
        else
        {
            auto it2 = this->availableBuffersIndices.begin();
            bufferIndex = *it2;
            this->availableBuffersIndices.erase(it2);
            Serializer &serializer = this->buffers[bufferIndex];
            serializer.reset();
            serializer.insert(static_cast<size_t>(0));
        }
    }
    else
    {
        bufferIndex = it->second;
    }

    this->ranksSendBuffers[rank] = bufferIndex;

    Serializer &serializer = this->buffers[bufferIndex];
    size_t bytes = 0;
    serializer.extract(bytes, 0);
    bytes += serializer.insert(value);
    serializer.hardSet(0, bytes);

    assert(bytes <= this->buffersSize);

    if(this->ShouldSend(rank))
    {
        this->Dispatch(rank);
    }
}

template<typename T>
bool BuffersManager<T>::ShouldSend(rank_t rank)
{
    const auto it = this->ranksSendBuffers.find(rank);
    if(it == this->ranksSendBuffers.end())
    {
        return false;
    }
    size_t bufferIndex = it->second;
    size_t bufferSize;
    this->buffers[bufferIndex].extract(bufferSize, 0);
    if(bufferSize == sizeof(size_t))
    {
        return false;
    }
    if(bufferSize >= this->minSizeToDispatch)
    {
        if(this->TraceSink != nullptr)
        {
            ++this->TraceSink->DispatchBySize; // every true return is followed by Dispatch(rank)
        }
        return true;
    }
    if(this->buffersToCycles[rank] >= this->minCyclesToDispatch)
    {
        if(this->TraceSink != nullptr)
        {
            ++this->TraceSink->DispatchByTimer;
        }
        return true;
    }
    this->buffersToCycles[rank] += 1;
    return false;
}

template<typename T>
void BuffersManager<T>::Dispatch(rank_t rank)
{
    assert(this->ranksSendBuffers.find(rank) != this->ranksSendBuffers.end());
    size_t bufferIndex = this->ranksSendBuffers.at(rank);

    assert(this->buffers[bufferIndex].size() >= sizeof(size_t));
    assert(this->buffers[bufferIndex].size() <= this->buffersSize);
    if(this->TraceSink != nullptr)
    {
        ++this->TraceSink->MessagesSent;
        this->TraceSink->BytesSent += this->buffers[bufferIndex].size();
    }
    MPI_Request &request = this->sendRequests.emplace_back(MPI_REQUEST_NULL);
    this->sendBuffersByRequests[this->sendRequests.size() - 1] = bufferIndex;
    MPI_Issend(this->buffers[bufferIndex].getData(), this->buffers[bufferIndex].size(), MPI_BYTE, rank, this->tag, this->comm, &request);
    this->activeSendRequests++;
    this->buffersToCycles[rank] = 0;
    this->ranksSendBuffers.erase(rank);
    this->sendCounters[rank]++;
    this->sendCounter++;
}

template<typename T>
void BuffersManager<T>::CleanSendRequests(void)
{
    static std::vector<int> array_of_indices;
    static std::vector<MPI_Status> array_of_statuses;
    if(this->sendRequests.empty())
    {
        return;
    }

    if(array_of_indices.size() != this->sendRequests.size())
    {
        array_of_indices.resize(this->sendRequests.size());
    }
    if(array_of_statuses.size() != this->sendRequests.size())
    {
        array_of_statuses.resize(this->sendRequests.size());
    }

    int outcount;
    MPI_Testsome(this->sendRequests.size(), this->sendRequests.data(), &outcount, array_of_indices.data(), array_of_statuses.data());

    for(int i = outcount - 1; i >= 0; i--)
    {
        int requestNum = array_of_indices[i];
        MPI_Request &request = this->sendRequests.at(requestNum);
        size_t bufferIndex = this->sendBuffersByRequests.at(requestNum);
        this->availableBuffersIndices.insert(bufferIndex);
        this->sendBuffersByRequests.erase(requestNum);

        size_t otherRequestNumber = this->sendRequests.size() - 1;
        if(otherRequestNumber != requestNum)
        {
            this->sendBuffersByRequests[requestNum] = this->sendBuffersByRequests.at(otherRequestNumber);
            this->sendBuffersByRequests.erase(otherRequestNumber);
        }
        std::swap(request, this->sendRequests.back());
        this->sendRequests.pop_back();

        this->activeSendRequests--;
    }
}

template<typename T>
void BuffersManager<T>::HandleIncomingOutcoming(void)
{
    std::vector<rank_t> ranksToDispatch;
    ranksToDispatch.reserve(this->ranksSendBuffers.size());
    for(const auto &it : this->ranksSendBuffers)
    {
        rank_t rank = it.first;
        if(this->ShouldSend(rank))
        {
            ranksToDispatch.push_back(rank);
        }
    }
    for(rank_t rank : ranksToDispatch)
    {
        this->Dispatch(rank);
    }
    this->Receive();
    this->CleanSendRequests();
}

#endif // __WITH_MPI

#endif // MPI_UTILS_BUFFERS_MANAGER_HPP
