#pragma once

#include <type_traits>

#include "factorial.hpp"

namespace acd
{
template <typename type, type n, type k>
struct combination_t : std::integral_constant<type, factorial_v<type, n> / (factorial_v<type, k> * factorial_v<type, n - k>)> {};

template <typename type, type n, type k>
inline constexpr type combination_v = combination_t<type, n, k>::value;

template <typename type>
constexpr type combination(const type n, const type k)
{
  return factorial<type>(n) / (factorial<type>(k) * factorial<type>(n - k));
}
}
