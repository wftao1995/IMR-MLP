// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once 

#include <cstddef>
#include <string_view>
#define __max(a, b) (((a) > (b)) ? (a) : (b))
#define __min(a, b) (((a) < (b)) ? (a) : (b))
#define __countof(a) (sizeof(a)/sizeof(*a))
#define __delta(a, b) (((a) == (b)) ? 1 : 0)

template <typename T>
constexpr std::string_view type_name() {
#ifdef __clang__
    constexpr std::string_view prefix = "T = ";
    constexpr std::string_view func = __PRETTY_FUNCTION__;
#elif defined(__GNUC__)
    constexpr std::string_view prefix = "with T = ";
    constexpr std::string_view func = __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
    constexpr std::string_view prefix = "type_name<";
    constexpr std::string_view func = __FUNCSIG__;
#endif
    const auto start = func.find(prefix) + prefix.size();
    const auto end = func.find_last_of(">]");
    return func.substr(start, end - start);
}

template<typename T>
constexpr int _countof(T t)
{
    return sizeof(t)/sizeof(*t);
}
template<typename T, size_t N>
constexpr int _length(const T(&)[N])
{
    return N;
}
template<typename T>
constexpr T _max(T a, T b)
{
    return a > b ? a : b;
}
template<typename T>
constexpr T _min(T a, T b)
{
    return a < b ? a : b;
}

// Math functions at compile time
template<typename T>
constexpr T _factorial(T a)
{
    return a == 1 ? 1 : a * _factorial(a - 1);
}

#define __is_close(a,b,tol) std::fabs(a - b) < tol

#define BOHR_RADIUS 0.529177210903
#define INV_SQU_BOHR_RADIUS 3.571064826093132
#define INV_BOHR_RADIUS 1.88972612462577
#define EV_TO_HARTREE 0.03674930813664888
#define HARTREE_TO_EV 27.211396641308
#define A_TO_BOHR 1.88972612462577
#define BOHR_TO_A 0.529177210903
#define COMPLEX_ONE (1.0 + 0.0i)
#define COMPLEX_ZERO (0.0 + 0.0i)
#define COMPLEX_IMG (0.0 + 1.0i)
#define COMPLEX_DIA (1.0 + 1.0i)

#include <iostream>
#include <vector>
#define _DEBUG_OUTPUT_(a) do { std::cout << "Debug " << __FILE__ << " line: " << __LINE__ << " " << "symbol: " << #a << " " << (a) << std::endl; } while(0);
#define _DEBUG_VECTOR_(a) do { std::cout << "Debug " << __FILE__ << " line: " << __LINE__ << " " << "symbol: " << #a << std::endl; for(auto &it : (a)) { std::cout << " " << it; }; std::cout << std::endl; } while(0);
