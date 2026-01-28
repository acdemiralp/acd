#pragma once

#include <type_traits>

namespace acd
{
// Note: These functions work best with unsigned integral types.
// Using signed types may lead to unexpected behavior with negative values.
template<typename type>
constexpr bool is_power_of_two(const type value)
{
  static_assert(std::is_integral_v<type>, "is_power_of_two requires integral types");
  
  return value > 0 && (value & (value - 1)) == 0;
}

// Returns the smallest power of two that is greater than or equal to value.
// Note: Does not check for overflow. If value is too large, result will be incorrect.
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

// Returns the largest power of two that is less than or equal to value.
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
