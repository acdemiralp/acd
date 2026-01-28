#pragma once

namespace acd
{
template<typename type>
constexpr int sign(const type value)
{
  return (type(0) < value) - (value < type(0));
}
}
