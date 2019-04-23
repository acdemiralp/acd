#ifndef ACD_DISTRIBUTIONS_UNIFORM_MULTIVARIATE_DISTRIBUTION_HPP
#define ACD_DISTRIBUTIONS_UNIFORM_MULTIVARIATE_DISTRIBUTION_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <istream>
#include <random>
#include <utility>

namespace acd
{
// Produces random points, uniformly distributed in an N-dimensional space.
// Satisfies https://en.cppreference.com/w/cpp/named_req/RandomNumberDistribution hence fully compatible with <random>.
template<typename type = std::array<float, 3>, typename underlying_distribution = std::uniform_real_distribution<typename type::value_type>>
class uniform_multivariate_distribution
{
public:
  using result_type         = type;
  using result_element_type = typename result_type::value_type;
  using scalar_distribution = underlying_distribution;

  static constexpr std::size_t result_element_count = typename std::tuple_size<result_type>::value;

  struct param_type
  {
  public:
    using distribution_type = uniform_multivariate_distribution;

    explicit param_type   (const std::array           <result_element_type, 2>&                                   bounds) : bounds_()
    {
      bounds_.fill(bounds);
      set_distributions();
    }
    explicit param_type   (const std::initializer_list<result_element_type>&                                      bounds) : bounds_()
    {
      std::array<result_element_type, 2> array;
      std::copy(bounds.begin(), bounds.end(), array.begin());
      bounds_.fill(array);
      set_distributions();
    }
    explicit param_type   (const std::array           <std::array<result_element_type, 2>, result_element_count>& bounds) : bounds_(bounds)
    {
      set_distributions();
    }                     
    explicit param_type   (const std::initializer_list<std::initializer_list<result_element_type>>&               bounds) : bounds_()
    {
      std::transform(bounds.begin(), bounds.end(), bounds_.begin(), [ ] (const std::initializer_list<result_element_type>& iteratee)
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
      return bounds_ == that.bounds_ && distributions_ == that.distributions_;
    }
    bool        operator!=(const param_type&  that) const
    {
      return !(*this == that);
    }

    const std::array<std::array<result_element_type, 2>, result_element_count>& bounds       () const
    {
      return bounds_;
    }
    const std::array<scalar_distribution               , result_element_count>& distributions() const
    {
      return distributions_;
    }

  protected:
    void set_distributions()
    {
      for (auto i = 0; i < bounds_.size(); ++i)
        distributions_[i] = scalar_distribution(bounds_[i][0], bounds_[i][1]);
    }

    std::array<std::array<result_element_type, 2>, result_element_count> bounds_       ;
    std::array<scalar_distribution               , result_element_count> distributions_;
  };

  explicit uniform_multivariate_distribution  (const std::array<result_element_type, 2>&                                   bounds = {result_element_type(0), result_element_type(1)}) 
  : parameters_(bounds)
  {
  
  }
  explicit uniform_multivariate_distribution  (const std::initializer_list<result_element_type>&                           bounds) 
  : parameters_(bounds)
  {

  }
  explicit uniform_multivariate_distribution  (const std::array<std::array<result_element_type, 2>, result_element_count>& bounds) 
  : parameters_(bounds)
  {
  
  }
  explicit uniform_multivariate_distribution  (const std::initializer_list<std::initializer_list<result_element_type>>&    bounds) 
  : parameters_(bounds)
  {

  }
  explicit uniform_multivariate_distribution  (const param_type& parameters)
  : parameters_(parameters)
  {                                        
                                           
  }                                        
  uniform_multivariate_distribution           (const uniform_multivariate_distribution&  that) = default;
  uniform_multivariate_distribution           (      uniform_multivariate_distribution&& temp) = default;
 ~uniform_multivariate_distribution           ()                                               = default;
  uniform_multivariate_distribution& operator=(const uniform_multivariate_distribution&  that) = default;
  uniform_multivariate_distribution& operator=(      uniform_multivariate_distribution&& temp) = default;
  
  void        reset            ()
  {
    // Intentionally blank.
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
  result_type operator()       (engine& engine) const
  {
    return evaluate(engine, parameters_);
  }
  template<typename engine>
  result_type operator()       (engine& engine, const param_type& parameters) const
  {
    return evaluate(engine, parameters);
  } 

  result_type (min)            () const
  {
    result_type min;
    auto& _bounds = bounds();
    for (auto i = 0; i < _bounds.size(); ++i)
      min[i] = _bounds[i][0];
    return min;
  }
  result_type (max)            () const
  {
    result_type max;
    auto& _bounds = bounds();
    for (auto i = 0; i < _bounds.size(); ++i)
      max[i] = _bounds[i][1];
    return max;
  }
  
  const std::array<std::array<result_element_type, 2>, result_element_count>& bounds       () const
  {
    return parameters_.bounds       ();
  }
  const std::array<scalar_distribution               , result_element_count>& distributions() const
  {
    return parameters_.distributions();
  }

protected:
  template<typename engine>
  result_type evaluate         (engine& engine, const param_type& parameters) const
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
bool operator==(const uniform_multivariate_distribution<type>& lhs, const uniform_multivariate_distribution<type>& rhs)
{
  return lhs.param() == rhs.param();
}
template<typename type>
bool operator!=(const uniform_multivariate_distribution<type>& lhs, const uniform_multivariate_distribution<type>& rhs)
{
  return !(lhs == rhs);
}
template<typename stream_type, typename stream_traits, typename type>
std::basic_ostream<stream_type, stream_traits>& operator<<(std::basic_ostream<stream_type, stream_traits>& stream, const uniform_multivariate_distribution<type>& distribution)
{
  auto& bounds = distribution.bounds();
  stream.write(bounds.data(), bounds.size());
  return stream;
}
template<typename stream_type, typename stream_traits, typename type>
std::basic_istream<stream_type, stream_traits>& operator>>(std::basic_istream<stream_type, stream_traits>& stream,       uniform_multivariate_distribution<type>& distribution)
{
  std::array<std::array<typename uniform_multivariate_distribution<type>::result_element_type, 2>, uniform_multivariate_distribution<type>::result_element_count> bounds;
  stream.read(bounds.data(), bounds.size());
  distribution.param(bounds);
  return stream;
}
}

#endif