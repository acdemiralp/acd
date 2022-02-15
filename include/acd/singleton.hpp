#ifndef ACD_SINGLETON_HPP
#define ACD_SINGLETON_HPP

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include <utility>

namespace acd
{
template <typename derived>
class singleton
{
public:
  __host__ __device__ singleton           ()                       = default;
  __host__ __device__ singleton           (const singleton&  that) = delete ;
  __host__ __device__ singleton           (      singleton&& temp) = delete ;
  __host__ __device__ virtual ~singleton  ()                       = default;
  __host__ __device__ singleton& operator=(const singleton&  that) = delete ;
  __host__ __device__ singleton& operator=(      singleton&& temp) = delete ;

  template <typename... argument_types>
  __host__ __device__ static derived& instance(argument_types&&... arguments)
  {
    static derived instance(std::forward<argument_types>(arguments)...);
    return instance;
  }
};
}

#endif