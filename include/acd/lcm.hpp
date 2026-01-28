#pragma once

#include <type_traits>

#include "gcd.hpp"

namespace acd
{
// Computes least common multiple. Note: can overflow for large values.
// For negative inputs, the result follows mathematical convention (always positive).
template<typename type>
constexpr type lcm(type a, type b)
{
  static_assert(std::is_integral_v<type>, "lcm requires integral types");
  
  if (a == 0 || b == 0)
    return 0;
  
  // Handle negative values by taking absolute value
  if constexpr (std::is_signed_v<type>)
  {
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
  }
  
  // Use (a / gcd) * b instead of (a * b) / gcd to reduce overflow risk
  return (a / gcd(a, b)) * b;
}
}
