# cpp-MPI-utils

A lightweight C++17 header-only library for MPI communication utilities.

## Overview

cpp-MPI-utils provides type-safe, serialization-based MPI communication patterns for C++ applications. It wraps common point-to-point, all-to-all, and collective operations behind template functions that work with trivial types, STL containers, and user-defined types implementing the `Serializable` interface.

The library is designed to be included directly in a project with no separate compilation step.

## Features

- **Serialization framework** — `Serializer` and `Serializable` for byte-buffer packing of scalars, strings, pairs, vectors, and custom types
- **Type-safe exchange** — point-to-point send/receive with automatic serialization (`MPI_exchange_data`, `MPI_exchange_data_indexed`)
- **All-to-all patterns** — dense and sparse all-to-all, exchange by rank correspondents, and ownership-based redistribution
- **Collectives** — broadcast, gather, scatter, and all-gather of serialized data (`MPI_Bcast_serializable`, `MPI_Gatherv_serializable`, `MPI_All_cast`, `MPI_Spread`)
- **Reduce utilities** — min/max location reductions (`MPI_Min_loc`, `MPI_Max_loc`)
- **Indexed exchange** — request specific elements from remote ranks (`MPI_Ask_data`, `MPI_exchange_data_indexed`)
- **High-level exchange** — `dataExchange` routes items to owner ranks and collects replies
- **Error handling** — structured exceptions via `MpiUtilsError` with contextual key–value diagnostics

## Requirements

- C++17 or later
- An MPI implementation (Open MPI, MPICH, Intel MPI, etc.)
- CMake 3.14+ (optional, for CMake integration)

## Usage

Add the directory containing the `mpi_utils/` folder to your include path, or link against the provided CMake interface target:

```cmake
add_subdirectory(path/to/cpp-MPI-utils)
target_link_libraries(my_target PRIVATE mpi_utils)
```

Include headers with the `mpi_utils/` prefix:

```cpp
#include <mpi_utils/mpi_commands.hpp>
#include <mpi_utils/serialize/Serializer.hpp>
```

### Example: exchange data between ranks

```cpp
#include <mpi.h>
#include <vector>
#include <mpi_utils/mpi_exchange.hpp>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Send local payload to every other rank
    std::vector<int> correspondents;
    for (int r = 0; r < size; ++r)
        if (r != rank) correspondents.push_back(r);

    std::vector<std::vector<int>> outbound(correspondents.size(),
                                            {rank, rank * 10});

    auto received = MPI_exchange_data(correspondents, outbound);

    // received[i] holds the vector sent back from correspondents[i]

    MPI_Finalize();
    return 0;
}
```

For custom types, inherit from `Serializable` and implement `dump` / `load`; the `Serializer` will dispatch automatically via `is_serializable`.

## API Reference

| Header | Purpose |
|--------|---------|
| `mpi_utils/mpi_commands.hpp` | Umbrella header; includes exchange, all-to-all, collectives, and reduce |
| `mpi_utils/mpi_exchange.hpp` | `MPI_exchange_data`, `MPI_exchange_data_indexed` — pairwise serialized exchange |
| `mpi_utils/mpi_alltoall.hpp` | All-to-all (dense/sparse), exchange by ownership/rank, `MPI_Ask_data`, `MPI_Distribute` |
| `mpi_utils/mpi_collectives.hpp` | `MPI_All_cast`, `MPI_Bcast_serializable`, `MPI_Gatherv_serializable`, `MPI_Spread` |
| `mpi_utils/mpi_reduce.hpp` | `MPI_Min_loc`, `MPI_Max_loc` (scalar and vector overloads) |
| `mpi_utils/exchange.hpp` | `dataExchange` — route items to owner ranks and gather responses |
| `mpi_utils/serialize/Serializer.hpp` | Byte-buffer serialization for supported types |
| `mpi_utils/serialize/Serializable.hpp` | Base class and trait for user-defined serializable types |
| `mpi_utils/MpiUtilsError.hpp` | Exception type with structured diagnostic entries |
| `mpi_utils/types.h` | Common type aliases (`rank_t`) and template utilities |
| `mpi_utils/MPI_complex_dtype.hpp` | Extension point for native MPI datatypes on specialized types |

## Error Handling

Operations validate inputs (rank bounds, buffer sizes, communicator consistency) and throw `MpiUtilsError`, a subclass of `std::runtime_error`. Errors carry a primary message plus optional named context entries accessible via `getEntries()`:

```cpp
try {
    auto result = MPI_Exchange_by_ownership(data, ownerFn, comm);
} catch (const MpiUtilsError& e) {
    std::cerr << e.what() << std::endl;
    for (const auto& [name, value] : e.getEntries())
        std::cerr << "  " << name << ": " << value << std::endl;
}
```

## License

See the LICENSE file in the repository root.
