#ifndef ACD_CONVERT_COORDINATES_HPP
#define ACD_CONVERT_COORDINATES_HPP

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include <cmath>

namespace acd
{
template<typename type>
__host__ __device__ type to_spherical(const type& cartesian)
{
  type spherical;
  spherical[0] = std::sqrt (std::pow(cartesian[0], 2) + std::pow(cartesian[1], 2) + std::pow(cartesian[2], 2));
  spherical[1] = std::atan2(cartesian[1] , cartesian[0]);
  spherical[2] = std::acos (cartesian[2] / spherical[0]);
  return spherical;
}
template<typename type>
__host__ __device__ type to_cartesian(const type& spherical)
{
  type cartesian;
  cartesian[0] = spherical[0] * std::cos(spherical[1]) * std::sin(spherical[2]);
  cartesian[1] = spherical[0] * std::sin(spherical[1]) * std::sin(spherical[2]);
  cartesian[2] = spherical[0] * std::cos(spherical[2]);
  return cartesian;
}
}

#endif
