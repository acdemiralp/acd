#pragma once

#include <type_traits>

#include "gcd.hpp"

namespace acd
{
template<typename type>
constexpr type lcm(const type a, const type b)
{
  static_assert(std::is_integral_v<type>, "lcm requires integral types");
  
  if (a == 0 || b == 0)
    return 0;
  
  return (a / gcd(a, b)) * b;
}
}
