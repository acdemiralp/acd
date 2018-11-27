#ifndef ACD_INTERPOLATION_HPP
#define ACD_INTERPOLATION_HPP

#include <algorithm>
#include <cmath>
#include <numeric>

namespace acd
{
template<typename type, typename weight_type>
type lerp (const type& a, const type& b, weight_type w) 
{
  type result;
  std::transform(a.begin(), a.end(), b.begin(), result.begin(), [&] (const auto a_element, const auto b_element)
  {
    return (weight_type(1) - w) * a_element + w * b_element;
  });
  return result;
}
template<typename type, typename weight_type>
type slerp(const type& a, const type& b, weight_type w)
{
  type result;
  auto o            = std::acos(std::inner_product(a.begin(), a.end(), b.begin(), 0));
  auto a_precompute = std::sin((weight_type(1) - w) * o) / std::sin(o);
  auto b_precompute = std::sin(                  w  * o) / std::sin(o);
  std::transform(a.begin(), a.end(), b.begin(), result.begin(), [&] (const auto a_element, const auto b_element)
  {
    return a_precompute * a_element + b_precompute * b_element;
  });
  return result;
}
}

#endif
