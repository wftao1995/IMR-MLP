// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef _IMRMLP_COMBINATORICS
#define _IMRMLP_COMBINATORICS
#include <cmath>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#ifdef IMRMLP_DEBUG
#include <iostream>
#endif
/**
 * Here all functions are unsafe, use boost version for safe but slower implementation
 * **/
#define MAX_PREDEFINED_FRACTORIAL 170
#define MAX_PREDEFINED_GAMMA_HALF 340
namespace Combinatorics
{
    // very easy to OVERFLOW!
    // DO NOT USE THIS UNLESS YOU KNOW WHAT YOU ARE DONIG!
    size_t factorial_int(size_t n);
    extern const double fact_table[171];
    extern const double gamma_half_table[343];
    // note that n! = gamma(n+1)
    double factorial(int n);
    // returns gamma(n/2), note n!=0, start from 1/2, then 2/2, ...
    double gamma_half(int n);
    // Integer-valued inputs, 0 <= m <= n. Integral results are exact or throw
    // overflow_error; floating results are rounded to R and must remain finite.
    // Floating input types remain accepted only for finite integer values.
    namespace detail {
    template<class T> std::uintmax_t binomial_input(T value) {
        static_assert(std::is_arithmetic_v<T>, "binomial inputs must be arithmetic");
        if constexpr(std::is_floating_point_v<T>) {
            if(!std::isfinite(value) || value<0 || std::trunc(value)!=value)
                throw std::invalid_argument("binomial requires nonnegative integer inputs");
            // max+1 is an exact power of two, unlike uintmax_t's rounded max.
            if(static_cast<long double>(value)>=std::ldexp(1.L,std::numeric_limits<std::uintmax_t>::digits))
                throw std::overflow_error("binomial input exceeds uintmax_t");
        } else if constexpr(std::is_signed_v<T>) {
            if(value<0) throw std::invalid_argument("binomial requires nonnegative integer inputs");
        }
        return static_cast<std::uintmax_t>(value);
    }
    }
    template <typename R, typename T1, typename T2>
    R binomial(T1 n_input, T2 m_input)
    {
        static_assert(std::is_arithmetic_v<R> && !std::is_same_v<R,bool>,
                      "binomial result must be an integral or floating numeric type");
        const auto n=detail::binomial_input(n_input);
        auto m=detail::binomial_input(m_input);
        if(m>n) throw std::invalid_argument("binomial requires m <= n");
        m=std::min(m,n-m);
        if constexpr(std::is_integral_v<R>) {
            std::uintmax_t result=1;
            const auto limit=static_cast<std::uintmax_t>(std::numeric_limits<R>::max());
            for(std::uintmax_t i=1;i<=m;++i) {
                auto numerator=n-m+i, denominator=i;
                auto factor=std::gcd(numerator,denominator);
                numerator/=factor;denominator/=factor;
                factor=std::gcd(result,denominator);
                result/=factor;denominator/=factor;
                // Exact binomial recurrence leaves no denominator after GCD.
                if(denominator!=1 || result>limit/numerator)
                    throw std::overflow_error("binomial result exceeds its integral type");
                result*=numerator;
            }
            return static_cast<R>(result);
        } else {
            long double result=1;
            const long double limit=std::numeric_limits<R>::max();
            for(std::uintmax_t i=1;i<=m;++i) {
                const long double factor=static_cast<long double>(n-m+i)/i;
                if(result>limit/factor)
                    throw std::overflow_error("binomial result exceeds its floating type");
                result*=factor;
                if(!std::isfinite(result) || result>limit)
                    throw std::overflow_error("binomial result exceeds its floating type");
            }
            return static_cast<R>(result);
        }
    }
    // Steinhaus–Johnson–Trotter algorithm, use Johnson 1963.
    class ElementaryPermutation
    {
    private:
        size_t *data = nullptr;
        size_t psize;
        int *direct = nullptr; // 0 left, 1 right

    public:
        ElementaryPermutation(size_t n) : psize(n)
        {
            data = new size_t[n];
            direct = new int[n];
            for (size_t i = 0; i < n; ++i)
            {
                data[i] = i;
            }
            memset(direct, 0, psize * sizeof(int));
        }
        ~ElementaryPermutation()
        {
            delete[] data;
            delete[] direct;
        }
        // even(1) or odd(-1)
        // The permutation is odd if and only if this factorization contains an odd number of even-length cycles.
        int parity()
        {
            int even = 0;
            int *mark = new int[psize]{0};
            for (size_t i = 0; i < psize; ++i)
            {
                if (mark[i] == 0)
                {
                    size_t p1 = data[i];
                    size_t leg = 1;
                    mark[i] = 1;
                    while (mark[p1] != 1)
                    {
                        mark[p1] = 1;
                        p1 = data[p1];
                        leg++;
                    }
                    if (leg % 2 == 0)
                    {
                        even++;
                    }
                }
            }
            delete[] mark;
            if (even % 2 == 1)
                return -1;
            return 1;
        }

        bool is_mobile(size_t n)
        {
            if (direct[n] == 0)
            {
                if (n == 0)
                    return false;
                if (data[n] > data[n - 1])
                    return true;
                return false;
            }
            else
            {
                if (n == psize - 1)
                    return false;
                if (data[n] > data[n + 1])
                    return true;
                return false;
            }
            return false; // just to avoid compile-time warning
        }
        /**
         * A directed integer is said to be mobile if it is greater than its 
         * immediate neighbor in the direction it is looking at.
         * 1. find the largest mobile integer k
         * 2. swap k and its neighbor in its direction
         * 3. reverse direction of all elements whose value is greater than k
         * if no mobile found, return <0,0>
         * */
        std::pair<size_t, size_t> next() // return swaps
        {
            size_t mob = 0;
            size_t site = 0;
            // setp 1
            for (size_t i = 0; i < psize; ++i)
            {
                if (is_mobile(i))
                    if (mob <= data[i])
                    {
                        mob = data[i];
                        site = i;
                    }
            }
            if (mob == 0 && site == 0)
                return std::pair<size_t, size_t>(0, 0);
            // step 2
            std::pair<size_t, size_t> ret;
            ret.first = site;
            if (direct[site] == 0)
            {
                std::swap(data[site - 1], data[site]);
                std::swap(direct[site - 1], direct[site]);
                ret.second = site - 1;
            }
            else
            {
                std::swap(data[site + 1], data[site]);
                std::swap(direct[site + 1], direct[site]);
                ret.second = site + 1;
            }
            // step 3
            for (size_t i = 0; i < psize; ++i)
            {
                if (data[i] > mob)
                {
                    direct[i] = 1 - direct[i];
                }
            }
            return ret;
        }

        size_t operator[](size_t id) const
        {
            return data[id % psize];
        }
    };

} // namespace Combinatorics

#endif
