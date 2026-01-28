#pragma once

#include <algorithm>

namespace acd
{
template<typename type>
constexpr type clamp(const type value, const type min, const type max)
{
  return std::min(std::max(value, min), max);
}
}
