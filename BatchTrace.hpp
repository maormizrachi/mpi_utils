#ifndef MPI_UTILS_BATCH_TRACE_HPP
#define MPI_UTILS_BATCH_TRACE_HPP

#include <cstddef>
#include <cstdint>

// Diagnostics only. Per-rank counters and wall seconds of the query machinery, written while a caller has attached a
// record (BuffersManager::SetTrace, QueryAgent::SetBatchTrace) and untouched otherwise. Recording never sends,
// receives, reduces or prints anything and never changes a message, its content or its order: it only reads clocks,
// local counters and the status of messages that were already received.

// One BuffersManager over its life.
struct BuffersManagerTrace {
    std::size_t DispatchBySize = 0;   // buffer reached the size threshold
    std::size_t DispatchByTimer = 0;  // buffer waited minCyclesToDispatch progress calls
    std::size_t DispatchByFlush = 0;  // DispatchAll (MADVORO_FLUSH_QUERY_TAIL)
    std::size_t MessagesSent = 0;
    std::size_t BytesSent = 0;
    std::size_t MessagesReceived = 0;
    std::size_t BytesReceived = 0;
};

// Every QueryBatchTracePollSamplePeriod-th progress pass of a query agent (by a counter kept over the agent's life) is
// timed and classified afterwards.
constexpr std::uint64_t QueryBatchTracePollSamplePeriod = 64;

// One BuffersManagerQueryAgent::runBatch; reset at the start of every batch. A progress pass ("poll") is one iteration
// of the batch loop. It is productive when, during it, a message was received, a send completed, an incoming query
// was answered, an answer was consumed or the termination state advanced; otherwise it is empty. Submit polls are the
// passes that still submit a query of this rank, drain polls the passes after the last one.
struct QueryBatchTrace {
    double ConstructSeconds = 0;           // both BuffersManagers (receive buffers allocated and zero-filled)
    double ConstructorBarrierSeconds = 0;  // AmountManager construction, i.e. its MPI_Barrier
    double SubmitSeconds = 0;              // batch loop until the last query of this rank was submitted
    double DrainSeconds = 0;               // rest of the batch loop, until termination
    double TeardownSeconds = 0;            // DestroyBuffersManagerPair (reduce-scatter) and AmountManager release
    double FinalBarrierSeconds = 0;        // the closing MPI_Barrier
    double RearrangeSeconds = 0;           // receive-peer list update and result rearrangement
    double ServeSeconds = 0;               // answering incoming queries (timed only on passes that answer some)
    std::size_t SubQueriesSent = 0;        // fan-out: query copies sent to other ranks
    std::size_t QueriesAnswered = 0;       // incoming query copies answered
    std::size_t AnswersReceived = 0;       // answer records received
    std::size_t SubmitPolls = 0;
    std::size_t SubmitPollsEmpty = 0;
    std::size_t DrainPolls = 0;
    std::size_t DrainPollsEmpty = 0;
    // Sampled passes, [0] empty and [1] productive: count and wall seconds of the whole pass.
    std::size_t SubmitSampled[2] = {0, 0};
    double SubmitSampledSeconds[2] = {0, 0};
    double SubmitSampledQuerySeconds = 0;  // part of the sampled submit seconds spent submitting the query itself
    std::size_t DrainSampled[2] = {0, 0};
    double DrainSampledSeconds[2] = {0, 0};
    BuffersManagerTrace QueryManager;
    BuffersManagerTrace AnswerManager;
};

#endif // MPI_UTILS_BATCH_TRACE_HPP
