#ifndef ACD_DISTRIBUTIONS_NORMAL_MULTIVARIATE_DISTRIBUTION_HPP
#define ACD_DISTRIBUTIONS_NORMAL_MULTIVARIATE_DISTRIBUTION_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <istream>
#include <limits>
#include <random>
#include <utility>

namespace acd
{
// Produces random points, normally distributed in an N-dimensional space.
// Satisfies https://en.cppreference.com/w/cpp/named_req/RandomNumberDistribution hence fully compatible with <random>.
template<typename type = std::array<float, 3>>
class normal_multivariate_distribution
{
public:
  using result_type         = type;
  using result_element_type = typename result_type::value_type;

  static constexpr std::size_t result_element_count = typename std::tuple_size<result_type>::value;

  struct param_type
  {
  public:
    using distribution_type = normal_multivariate_distribution;

    explicit param_type   (const std::array           <result_element_type, 2>&                                   scalar_parameters) : scalar_parameters_()
    {
      scalar_parameters_.fill(scalar_parameters);
      set_distributions();
    }
    explicit param_type   (const std::initializer_list<result_element_type>&                                      scalar_parameters) : scalar_parameters_()
    {
      std::array<result_element_type, 2> array;
      std::copy(scalar_parameters.begin(), scalar_parameters.end(), array.begin());
      scalar_parameters_.fill(array);
      set_distributions();
    }
    explicit param_type   (const std::array           <std::array<result_element_type, 2>, result_element_count>& scalar_parameters) : scalar_parameters_(scalar_parameters)
    {
      set_distributions();
    }                     
    explicit param_type   (const std::initializer_list<std::initializer_list<result_element_type>>&               scalar_parameters) : scalar_parameters_()
    {
      std::transform(scalar_parameters.begin(), scalar_parameters.end(), scalar_parameters_.begin(), [ ] (const std::initializer_list<result_element_type>& iteratee)
      {
        std::array<result_element_type, 2> array;
        std::copy(iteratee.begin(), iteratee.end(), array.begin());
        return array;
      });
      set_distributions();
    }
    param_type            (const param_type&  that) = default;
    param_type            (      param_type&& temp) = default;
   ~param_type            ()                        = default;
    param_type& operator= (const param_type&  that) = default;
    param_type& operator= (      param_type&& temp) = default;
    bool        operator==(const param_type&  that) const
    {
      return scalar_parameters_ == that.scalar_parameters_ && distributions_ == that.distributions_;
    }
    bool        operator!=(const param_type&  that) const
    {
      return !(*this == that);
    }

    const std::array<std::array              <result_element_type, 2>, result_element_count>& scalar_parameters() const
    {
      return scalar_parameters_;
    }
          std::array<std::normal_distribution<result_element_type>   , result_element_count>& distributions    ()
    {
      return distributions_;
    }

  protected:
    void set_distributions()
    {
      for (auto i = 0; i < scalar_parameters_.size(); ++i)
        distributions_[i] = std::normal_distribution<result_element_type>(scalar_parameters_[i][0], scalar_parameters_[i][1]);
    }

    std::array<std::array              <result_element_type, 2>, result_element_count> scalar_parameters_;
    std::array<std::normal_distribution<result_element_type>   , result_element_count> distributions_    ;
  };

  explicit normal_multivariate_distribution  (const std::array<result_element_type, 2>&                                   scalar_parameters = {result_element_type(0), result_element_type(1)}) 
  : parameters_(scalar_parameters)
  {
  
  }
  explicit normal_multivariate_distribution  (const std::initializer_list<result_element_type>&                           scalar_parameters) 
  : parameters_(scalar_parameters)
  {

  }
  explicit normal_multivariate_distribution  (const std::array<std::array<result_element_type, 2>, result_element_count>& scalar_parameters) 
  : parameters_(scalar_parameters)
  {
  
  }
  explicit normal_multivariate_distribution  (const std::initializer_list<std::initializer_list<result_element_type>>&    scalar_parameters) 
  : parameters_(scalar_parameters)
  {

  }
  explicit normal_multivariate_distribution  (const param_type& parameters)
  : parameters_(parameters)
  {                                        
                                           
  }                                        
  normal_multivariate_distribution           (const normal_multivariate_distribution&  that) = default;
  normal_multivariate_distribution           (      normal_multivariate_distribution&& temp) = default;
 ~normal_multivariate_distribution           ()                                              = default;
  normal_multivariate_distribution& operator=(const normal_multivariate_distribution&  that) = default;
  normal_multivariate_distribution& operator=(      normal_multivariate_distribution&& temp) = default;
  
  void        reset            ()
  {
    for (auto& distribution : distributions())
      distribution->reset();
  }
              
  param_type  param            () const
  {
    return parameters_;
  }
  void        param            (const param_type& parameters)
  {
    parameters_ = parameters;
  }
  
  template<typename engine>
  result_type operator()       (engine& engine)
  {
    return evaluate(engine, parameters_);
  }
  template<typename engine>
  result_type operator()       (engine& engine, const param_type& parameters)
  {
    return evaluate(engine, parameters);
  } 

  result_type (min)            () const
  {
    result_type min;
    for (auto i = 0; i < result_element_count; ++i)
      min[i] = std::numeric_limits<result_element_type>::denorm_min();
    return min;
  }
  result_type (max)            () const
  {
    result_type max;
    for (auto i = 0; i < result_element_count; ++i)
      max[i] = std::numeric_limits<result_element_type>::max();
    return max;
  }
  
  const std::array<std::array              <result_element_type, 2>, result_element_count>& scalar_parameters() const
  {
    return parameters_.scalar_parameters();
  }
        std::array<std::normal_distribution<result_element_type>   , result_element_count>& distributions    ()
  {
    return parameters_.distributions    ();
  }

protected:
  template<typename engine>
  result_type evaluate         (engine& engine, const param_type& parameters)
  {
    result_type value;
    auto& _distributions = distributions();
    for (auto i = 0; i < _distributions.size(); ++i)
      value[i] = _distributions[i](engine);
    return value;
  }

  param_type parameters_;
};

template<typename type>
bool operator==(const normal_multivariate_distribution<type>& lhs, const normal_multivariate_distribution<type>& rhs)
{
  return lhs.param() == rhs.param();
}
template<typename type>
bool operator!=(const normal_multivariate_distribution<type>& lhs, const normal_multivariate_distribution<type>& rhs)
{
  return !(lhs == rhs);
}
template<typename stream_type, typename stream_traits, typename type>
std::basic_ostream<stream_type, stream_traits>& operator<<(std::basic_ostream<stream_type, stream_traits>& stream, const normal_multivariate_distribution<type>& distribution)
{
  auto& scalar_parameters = distribution.scalar_parameters();
  stream.write(scalar_parameters.data(), scalar_parameters.size());
  return stream;
}
template<typename stream_type, typename stream_traits, typename type>
std::basic_istream<stream_type, stream_traits>& operator>>(std::basic_istream<stream_type, stream_traits>& stream,       normal_multivariate_distribution<type>& distribution)
{
  std::array<std::array<typename normal_multivariate_distribution<type>::result_element_type, 2>, normal_multivariate_distribution<type>::result_element_count> scalar_parameters;
  stream.read(scalar_parameters.data(), scalar_parameters.size());
  distribution.param(scalar_parameters);
  return stream;
}
}

#endif