#pragma once

#define _USE_MATH_DEFINES

#include <math.h>

namespace acd
{
template<typename type>
constexpr type to_radians(type degrees)
{
  return degrees * type(M_PI / 180.0);
}
template<typename type>
constexpr type to_degrees(type radians)
{
  return radians * type(180.0 / M_PI);
}
}