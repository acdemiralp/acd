#ifndef ACD_DISTRIBUTIONS_UNIFORM_SPHERICAL_DISTRIBUTION_HPP
#define ACD_DISTRIBUTIONS_UNIFORM_SPHERICAL_DISTRIBUTION_HPP

#define _USE_MATH_DEFINES

#include <array>
#include <cmath>
#include <istream>
#include <random>
#include <type_traits>

namespace acd
{
// Produces random points, uniformly distributed on the surface of the unit 2-sphere/hemisphere.
// Satisfies https://en.cppreference.com/w/cpp/named_req/RandomNumberDistribution hence fully compatible with <random>.
// Supports Cartesian and spherical coordinates, based on equations 1,2,6,7,8 at http://mathworld.wolfram.com/SpherePointPicking.html.
// Spherical coordinates follow the radius-longitude-colatitude convention at http://mathworld.wolfram.com/SphericalCoordinates.html.
// Hemisphere range generates +Z samples in Cartesian coordinates and colatitude <= PI/2 samples in spherical coordinates.
template<typename type = std::array<float, 3>>
class uniform_spherical_distribution
{
public:
  enum class coordinate_system_type
  {
    cartesian,
    spherical
  };
  enum class range_type
  {
    sphere,
    hemisphere
  };

  using result_type         = type;
  using result_element_type = typename result_type::value_type;

  struct param_type
  {
  public:
    using distribution_type = uniform_spherical_distribution;

    explicit param_type   (const coordinate_system_type coordinate_system = coordinate_system_type::cartesian, const range_type range = range_type::sphere)
    : coordinate_system_(coordinate_system)
    , range_            (range)
    {
      if (coordinate_system_ == coordinate_system_type::cartesian)
      {
        u_distribution_ = std::uniform_real_distribution<result_element_type>(range_ == range_type::sphere ? -1 : 0, 1);
        v_distribution_ = std::uniform_real_distribution<result_element_type>(0, 2 * M_PI);
      }
      else
      {
        u_distribution_ = std::uniform_real_distribution<result_element_type>(0, 1);
        v_distribution_ = std::uniform_real_distribution<result_element_type>(range_ == range_type::sphere ? 0 : 0.5, 1);
      }
    }                                        
    param_type            (const param_type&  that) = default;
    param_type            (      param_type&& temp) = default;
   ~param_type            ()                        = default;
    param_type& operator= (const param_type&  that) = default;
    param_type& operator= (      param_type&& temp) = default;
    bool        operator==(const param_type&  that) const
    {
      return 
        coordinate_system_ == that.coordinate_system_ && 
        range_             == that.range_             &&
        u_distribution_    == that.u_distribution_    &&
        v_distribution_    == that.v_distribution_    ;
    }
    bool        operator!=(const param_type&  that) const
    {
      return !(*this == that);
    }

    coordinate_system_type                                     coordinate_system() const
    {
      return coordinate_system_;
    }
    range_type                                                 range            () const
    {
      return range_;
    }
    const std::uniform_real_distribution<result_element_type>& u_distribution   () const
    {
      return u_distribution_;
    }
    const std::uniform_real_distribution<result_element_type>& v_distribution   () const
    {
      return v_distribution_;
    }

  protected:
    coordinate_system_type                              coordinate_system_;
    range_type                                          range_            ;
    std::uniform_real_distribution<result_element_type> u_distribution_   ;
    std::uniform_real_distribution<result_element_type> v_distribution_   ;
  };

  explicit uniform_spherical_distribution  (const coordinate_system_type coordinate_system = coordinate_system_type::cartesian, const range_type range = range_type::sphere)
  : parameters_(coordinate_system, range)
  {

  }
  explicit uniform_spherical_distribution  (const param_type& parameters)
  : parameters_(parameters)
  {                                        
                                           
  }                                        
  uniform_spherical_distribution           (const uniform_spherical_distribution&  that) = default;
  uniform_spherical_distribution           (      uniform_spherical_distribution&& temp) = default;
 ~uniform_spherical_distribution           ()                                            = default;
  uniform_spherical_distribution& operator=(const uniform_spherical_distribution&  that) = default;
  uniform_spherical_distribution& operator=(      uniform_spherical_distribution&& temp) = default;
  
  void                   reset            ()
  {
    // Intentionally blank.
  }
                         
  param_type             param            () const
  {
    return parameters_;
  }
  void                   param            (const param_type& parameters)
  {
    parameters_ = parameters;
  }
  
  template<typename engine>
  result_type            operator()       (engine& engine) const
  {
    return evaluate(engine, parameters_);
  }
  template<typename engine>
  result_type            operator()       (engine& engine, const param_type& parameters) const
  {
    return evaluate(engine, parameters);
  } 

  result_type            (min)            () const
  {
    return parameters_.coordinate_system() == coordinate_system_type::cartesian 
      ? result_type
      {
        result_element_type(-1), 
        result_element_type(-1), 
        result_element_type(parameters_.range() == range_type::sphere ? -1 : 0)
      }
      : result_type
      {
        result_element_type(1), 
        result_element_type(0), 
        result_element_type(0)
      };
  }
  result_type            (max)            () const
  {
    return parameters_.coordinate_system() == coordinate_system_type::cartesian
      ? result_type
      {
        result_element_type(1), 
        result_element_type(1), 
        result_element_type(1)
      }
      : result_type
      { 
        result_element_type(1), 
        result_element_type(2 * M_PI), 
        result_element_type(parameters_.range() == range_type::sphere ? M_PI : M_PI / 2)
      };
  }

  coordinate_system_type coordinate_system() const
  {
    return parameters_.coordinate_system();
  }
  range_type             range            () const
  {
    return parameters_.range();
  }
  
protected:
  template<typename engine>
  result_type            evaluate         (engine& engine, const param_type& parameters) const
  {
    auto u = parameters.u_distribution()(engine);
    auto v = parameters.v_distribution()(engine);
    return parameters.coordinate_system() == coordinate_system_type::cartesian
      ? result_type 
      {
        result_element_type(std::sqrt(1 - std::pow(u, 2)) * std::cos(v)), 
        result_element_type(std::sqrt(1 - std::pow(u, 2)) * std::sin(v)), 
        result_element_type(u)
      }
      : result_type 
      {
        result_element_type(1), 
        result_element_type(2 * M_PI * u), 
        result_element_type(std::acos(2 * v - 1))
      };
  }

  param_type parameters_;
};

template<typename type>
bool operator==(const uniform_spherical_distribution<type>& lhs, const uniform_spherical_distribution<type>& rhs)
{
  return lhs.param() == rhs.param();
}
template<typename type>
bool operator!=(const uniform_spherical_distribution<type>& lhs, const uniform_spherical_distribution<type>& rhs)
{
  return !(lhs == rhs);
}
template<typename stream_type, typename stream_traits, typename type>
std::basic_ostream<stream_type, stream_traits>& operator<<(std::basic_ostream<stream_type, stream_traits>& stream, const uniform_spherical_distribution<type>& distribution)
{
  stream << static_cast<typename std::underlying_type<typename uniform_spherical_distribution<type>::coordinate_system_type>::type>(distribution.coordinate_system());
  stream << static_cast<typename std::underlying_type<typename uniform_spherical_distribution<type>::range_type            >::type>(distribution.range            ());
  return stream;
}
template<typename stream_type, typename stream_traits, typename type>
std::basic_istream<stream_type, stream_traits>& operator>>(std::basic_istream<stream_type, stream_traits>& stream,       uniform_spherical_distribution<type>& distribution)
{
  typename std::underlying_type<typename uniform_spherical_distribution<type>::coordinate_system_type>::type coordinate_system;
  typename std::underlying_type<typename uniform_spherical_distribution<type>::range_type>::type             range            ;
  stream >> coordinate_system;
  stream >> range            ;
  distribution.param(typename uniform_spherical_distribution<type>::param_type(
    static_cast<typename uniform_spherical_distribution<type>::coordinate_system_type>(coordinate_system), 
    static_cast<typename uniform_spherical_distribution<type>::range_type>            (range            )));
  return stream;
}
}

#endif