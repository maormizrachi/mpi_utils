#ifndef MPI_UTILS_TYPES_H
#define MPI_UTILS_TYPES_H

#include <type_traits>
#include <vector>

using rank_t = int;

template<typename Test, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref>: std::true_type {};

#endif // MPI_UTILS_TYPES_H
