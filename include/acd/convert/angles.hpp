#ifndef ACD_CONVERT_ANGLES_HPP
#define ACD_CONVERT_ANGLES_HPP

#define _USE_MATH_DEFINES
#include <math.h>

namespace acd
{
template<typename type>
type to_radians(type degrees)
{
  return degrees * type(M_PI / 180.0);
}
template<typename type>
type to_degrees(type radians)
{
  return radians * type(180.0 / M_PI);
}
}

#endif