#ifndef ACD_PERMUTE_FOR_HPP
#define ACD_PERMUTE_FOR_HPP

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

namespace acd
{
// Permutes the loop for(auto i = start, i < end; i+= step) over all dimensions. 
template<std::size_t dimensions>
void permute_for(
  const std::function<void(const std::array<std::size_t, dimensions>&)>& function, 
  const std::array<std::size_t, dimensions>&                             start   ,
  const std::array<std::size_t, dimensions>&                             end     ,
  const std::array<std::size_t, dimensions>&                             step    )
{
  std::function<void(std::array<std::size_t, dimensions>, std::size_t)> permute_for_internal = 
    [&] (std::array<std::size_t, dimensions> indices, std::size_t depth)
    {
      if (depth < dimensions)
      {
        for (auto i = start[depth]; i < end[depth]; i += step[depth])
        {
          indices[depth] = i;
          permute_for_internal(indices, depth + 1);
        }
      }
      else
        function(indices);
    };
  permute_for_internal({}, 0);
}

inline void permute_for(
  const std::function<void(const std::vector<std::size_t>&)>&            function, 
  const std::vector<std::size_t>&                                        start   ,
  const std::vector<std::size_t>&                                        end     ,
  const std::vector<std::size_t>&                                        step    )
{
  std::function<void(std::vector<std::size_t>, std::size_t)> permute_for_internal =
    [&] (std::vector<std::size_t> indices, std::size_t depth)
    {
      if (depth < start.size())
      {
        for (auto i = start[depth]; i < end[depth]; i += step[depth])
        {
          indices[depth] = i;
          permute_for_internal(indices, depth + 1);
        }
      }
      else
        function(indices);
    };
  permute_for_internal(std::vector<std::size_t>(start.size(), 0), 0);
}
}

#endif