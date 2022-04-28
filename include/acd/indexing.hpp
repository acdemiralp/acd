#pragma once

#include <cstdint>
#include <cstddef>

namespace acd
{
// Ducks [] and .size() on the type.
template <typename type>
constexpr type        unravel_index    (std::size_t index      , const type& dimensions)
{
  type subscripts;
  for (std::int64_t i = dimensions.size() - 1; i >= 0; --i)
  {
    subscripts[i] = index % dimensions[i];
    index = index / dimensions[i];
  }
  return subscripts;
}

// Ducks [] and .size() on the type.
template <typename type>
constexpr std::size_t ravel_multi_index(const type& multi_index, const type& dimensions)
{
  std::size_t index(0);
  for (std::size_t i = 0; i < dimensions.size(); ++i)
    index = index * dimensions[i] + multi_index[i];
  return index;
}
}