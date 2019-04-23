#ifndef ACD_PRIME_FACTORIZATION_HPP_
#define ACD_PRIME_FACTORIZATION_HPP_

#include <cmath>
#include <vector>

namespace acd
{
template<typename type>
std::vector<type> prime_factorize(type value)
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