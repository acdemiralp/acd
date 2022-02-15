#ifndef ACD_CONSTEXPR_FOR_HPP
#define ACD_CONSTEXPR_FOR_HPP

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace acd
{
template <auto begin, auto end, auto increment, typename function_type>
__host__ __device__ constexpr void constexpr_for      (function_type&& function)
{
  if constexpr (begin < end)
  {
    function(std::integral_constant<decltype(begin), begin>());
    constexpr_for<begin + increment, end, increment>(function);
  }
}
template <typename function_type, typename... argument_types>
__host__ __device__ constexpr void constexpr_for      (function_type&& function, argument_types&&... arguments)
{
  (function(std::forward<argument_types>(arguments)), ...);
}
template <typename function_type, typename tuple_type>
__host__ __device__ constexpr void constexpr_for_tuple(function_type&& function, tuple_type&& tuple)
{
  constexpr std::size_t count = std::tuple_size_v<std::decay_t<tuple_type>>;
  constexpr_for<static_cast<std::size_t>(0), count, static_cast<std::size_t>(1)>([&] (auto i)
  {
    function(std::get<i.value>(tuple));
  });
}
}

#endif
