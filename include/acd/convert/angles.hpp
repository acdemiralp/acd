#ifndef ACD_CONVERT_ANGLES_HPP
#define ACD_CONVERT_ANGLES_HPP

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#define _USE_MATH_DEFINES

#include <math.h>

namespace acd
{
template<typename type>
__host__ __device__ type to_radians(type degrees)
{
  return degrees * type(M_PI / 180.0);
}
template<typename type>
__host__ __device__ type to_degrees(type radians)
{
  return radians * type(180.0 / M_PI);
}
}

#endif