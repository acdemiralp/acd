#pragma once

#include <type_traits>

namespace acd
{
template<typename type>
constexpr type gcd(type a, type b)
{
  static_assert(std::is_integral_v<type>, "gcd requires integral types");
  
  // Handle negative values by taking absolute value
  if constexpr (std::is_signed_v<type>)
  {
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
  }
  
  while (b != 0)
  {
    const auto temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}
}
