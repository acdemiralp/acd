#pragma once

#include <type_traits>

namespace acd
{
template<typename type>
constexpr bool is_power_of_two(const type value)
{
  static_assert(std::is_integral_v<type>, "is_power_of_two requires integral types");
  
  return value > 0 && (value & (value - 1)) == 0;
}

template<typename type>
constexpr type next_power_of_two(type value)
{
  static_assert(std::is_integral_v<type>, "next_power_of_two requires integral types");
  
  if (value == 0)
    return 1;
  
  --value;
  for (auto i = 1; i < sizeof(type) * 8; i *= 2)
    value |= value >> i;
  
  return value + 1;
}

template<typename type>
constexpr type previous_power_of_two(type value)
{
  static_assert(std::is_integral_v<type>, "previous_power_of_two requires integral types");
  
  if (value == 0)
    return 0;
  
  for (auto i = 1; i < sizeof(type) * 8; i *= 2)
    value |= value >> i;
  
  return value - (value >> 1);
}
}
