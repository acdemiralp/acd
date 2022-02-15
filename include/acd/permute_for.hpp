#ifndef ACD_PERMUTE_FOR_HPP
#define ACD_PERMUTE_FOR_HPP

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include <cstddef>
#include <functional>

namespace acd
{
// Ducks [] and .size() on the type.
template <typename type>
__host__ __device__ void permute_for(
  const std::function<void(const type&)>& function, 
  const type&                             start   ,
  const type&                             end     ,
  const type&                             step    )
{
  std::function<void(type, std::size_t)> permute_for_internal =
    [&] (type indices, std::size_t depth)
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
  permute_for_internal(type(), 0);
}
}

#endif