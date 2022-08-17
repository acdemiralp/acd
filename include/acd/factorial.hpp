#pragma once

#include <type_traits>
#include <utility>

namespace acd
{
template <typename type, type n, typename = std::make_integer_sequence<type, n>>
struct factorial_t;
template <typename type, type n, type... sequence>
struct factorial_t<type, n, std::integer_sequence<type, sequence...>> : std::integral_constant<type, (... * (sequence + 1))> {};

template <typename type, type n>
inline constexpr type factorial_v = factorial_t<type, n>::value;

template <typename type>
constexpr type factorial(const type n)
{
  type result{1};
  for (auto i = type{2}; i <= n; ++i)
    result *= i;
  return result;
}
}
