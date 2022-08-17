#pragma once

#include <type_traits>

#include "factorial.hpp"

namespace acd
{
template <typename type, type n, type k>
struct binomial_coefficient_t : std::integral_constant<type, factorial_v<type, n> / (factorial_v<type, k> * factorial_v<type, n - k>)> {};

template <typename type, type n, type k>
inline constexpr type binomial_coefficient_v = binomial_coefficient_t<type, n, k>::value;

template <typename type>
constexpr type binomial_coefficient(const type n, const type k)
{
  return factorial<type>(n) / (factorial<type>(k) * factorial<type>(n - k));
}
}
