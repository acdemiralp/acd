#pragma once

#include <type_traits>

#include "gcd.hpp"

namespace acd
{
// Computes least common multiple. Note: can overflow for large values.
template<typename type>
constexpr type lcm(const type a, const type b)
{
  static_assert(std::is_integral_v<type>, "lcm requires integral types");
  
  if (a == 0 || b == 0)
    return 0;
  
  // Use (a / gcd) * b instead of (a * b) / gcd to reduce overflow risk
  return (a / gcd(a, b)) * b;
}
}
