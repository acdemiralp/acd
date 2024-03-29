#pragma once

#include <utility>

namespace acd
{
template <typename derived>
class singleton
{
public:
  singleton           ()                       = default;
  singleton           (const singleton&  that) = delete ;
  singleton           (      singleton&& temp) = delete ;
  virtual ~singleton  ()                       = default;
  singleton& operator=(const singleton&  that) = delete ;
  singleton& operator=(      singleton&& temp) = delete ;

  template <typename... argument_types>
  constexpr static derived& instance(argument_types&&... arguments)
  {
    static derived instance(std::forward<argument_types>(arguments)...);
    return instance;
  }
};
}