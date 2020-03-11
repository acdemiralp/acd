#ifndef ACD_INDEXING_HPP
#define ACD_INDEXING_HPP

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace acd
{
inline std::vector<std::size_t>            unravel_index    (std::size_t                         index      , std::vector<std::size_t>            size, const bool fortran_ordering = false)
{
  std::vector<std::size_t> subscripts(size.size(), 0);
  if (fortran_ordering)
    for (std::size_t  i = 0; i < size.size(); ++i)
    {
      subscripts[i] = index % size[i];
      index         = index / size[i];
    }
  else
    for (std::int64_t i = size.size() - 1; i >= 0; --i)
    {
      subscripts[i] = index % size[i];
      index         = index / size[i];
    }
  return subscripts;
}
inline std::size_t                         ravel_multi_index(std::vector<std::size_t>            multi_index, std::vector<std::size_t>            size, const bool fortran_ordering = false)
{
  std::size_t index(0);
  if (fortran_ordering)
    for (std::int64_t i = size.size() - 1; i >= 0; --i)
      index = index * size[i] + multi_index[i];
  else
    for (std::size_t  i = 0; i < size.size(); ++i)
      index = index * size[i] + multi_index[i];
  return index;
}

template <std::size_t dimensions>
inline std::array<std::size_t, dimensions> unravel_index    (std::size_t                         index      , std::array<std::size_t, dimensions> size, const bool fortran_ordering = false)
{
  std::array<std::size_t, dimensions> subscripts;
  subscripts.fill(0);
  if (fortran_ordering)
    for (std::size_t  i = 0; i < size.size(); ++i)
    {
      subscripts[i] = index % size[i];
      index         = index / size[i];
    }
  else
    for (std::int64_t i = size.size() - 1; i >= 0; --i)
    {
      subscripts[i] = index % size[i];
      index         = index / size[i];
    }
  return subscripts;
}
template <std::size_t dimensions>
inline std::size_t                         ravel_multi_index(std::array<std::size_t, dimensions> multi_index, std::array<std::size_t, dimensions> size, const bool fortran_ordering = false)
{
  std::size_t index(0);
  if (fortran_ordering)
    for (std::int64_t i = size.size() - 1; i >= 0; --i)
      index = index * size[i] + multi_index[i];
  else
    for (std::size_t  i = 0; i < size.size(); ++i)
      index = index * size[i] + multi_index[i];
  return index;
}
}

#endif
