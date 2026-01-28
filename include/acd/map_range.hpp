#pragma once

namespace acd
{
template<typename type>
constexpr type map_range(const type value, const type in_min, const type in_max, const type out_min, const type out_max)
{
  return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min);
}
}
