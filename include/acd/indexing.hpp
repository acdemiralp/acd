#ifndef ACD_INDEXING_HPP
#define ACD_INDEXING_HPP

#include <cstdint>
#include <cstddef>
#include <vector>

namespace acd
{
inline std::vector<std::size_t> unravel_index    (std::size_t              index      , std::vector<std::size_t> dimensions, const bool fortran_ordering = false)
{
  std::vector<std::size_t> subscripts(dimensions.size(), 0);
  if (fortran_ordering)
    for (std::size_t  i = 0; i < dimensions.size(); ++i)
    {
      subscripts[i] = index % dimensions[i];
      index         = index / dimensions[i];
    }
  else
    for (std::int64_t i = dimensions.size() - 1; i >= 0; --i)
    {
      subscripts[i] = index % dimensions[i];
      index         = index / dimensions[i];
    }
  return subscripts;
}
inline std::size_t              ravel_multi_index(std::vector<std::size_t> multi_index, std::vector<std::size_t> dimensions, const bool fortran_ordering = false)
{
  std::size_t index(0);
  if (fortran_ordering)
    for (std::int64_t i = dimensions.size() - 1; i >= 0; --i)
      index = index * dimensions[i] + multi_index[i];
  else
    for (std::size_t  i = 0; i < dimensions.size(); ++i)
      index = index * dimensions[i] + multi_index[i];
  return index;
}
}

#endif