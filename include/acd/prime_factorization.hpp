#ifndef ACD_PRIME_FACTORIZATION_HPP_
#define ACD_PRIME_FACTORIZATION_HPP_

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include <cmath>
#include <vector>

namespace acd
{
template<typename type>
__host__ __device__ std::vector<type> prime_factorize(type value)
{
  std::vector<type> prime_factors;

  type denominator = 2;
  while (std::pow(denominator, 2) <= value)
  {
    if (value % denominator == 0)
    {
      prime_factors.push_back(denominator);
      value /= denominator;
    }
    else
      ++denominator;
  }
  if (value > 1)
    prime_factors.push_back(value);

  return prime_factors;
}
}

#endif