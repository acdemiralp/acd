#ifndef ACD_DISTRIBUTIONS_VON_MISES_FISHER_DISTRIBUTION_HPP
#define ACD_DISTRIBUTIONS_VON_MISES_FISHER_DISTRIBUTION_HPP

#define _USE_MATH_DEFINES

#include <array>
#include <cmath>
#include <random>

namespace acd
{
// Produces random points, distributed on the surface of the unit 2-sphere/hemisphere.
// Satisfies https://en.cppreference.com/w/cpp/named_req/RandomNumberDistribution hence fully compatible with <random>.
// Based on chapter 3 of https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf. 
// Supports Cartesian and spherical coordinates.
// Spherical coordinates follow the radius-longitude-colatitude convention at http://mathworld.wolfram.com/SphericalCoordinates.html.
// Rotation matrix for orienting a vector to another vector is based on http://cs.brown.edu/research/pubs/pdfs/1999/Moller-1999-EBA.pdf.
template<typename type = std::array<float, 3>>
class von_mises_fisher_distribution
{
public:
  enum class coordinate_system_type
  {
    cartesian,
    spherical
  };

  using result_type         = type;
  using result_element_type = typename result_type::value_type;

  struct param_type
  {
  public:
    using distribution_type = von_mises_fisher_distribution;

    explicit param_type   (const result_type mean, const result_element_type concentration, const coordinate_system_type coordinate_system = coordinate_system_type::cartesian)
    : mean_             (mean             )
    , concentration_    (concentration    )
    , coordinate_system_(coordinate_system)
    {
      
    }
    param_type            (const param_type&  that) = default;
    param_type            (      param_type&& temp) = default;
   ~param_type            ()                        = default;
    param_type& operator= (const param_type&  that) = default;
    param_type& operator= (      param_type&& temp) = default;
    bool        operator==(const param_type&  that) const
    {
      return 
        mean_              == that.mean_              &&
        concentration_     == that.concentration_     &&
        coordinate_system_ == that.coordinate_system_ &&
        v_distribution_    == that.v_distribution_    &&
        w_distribution_    == that.w_distribution_    ;
    }
    bool        operator!=(const param_type&  that) const
    {
      return !(*this == that);
    }

    result_type                                                mean             () const
    {
      return mean_;
    }
    result_element_type                                        concentration    () const
    {
      return concentration_;
    }
    coordinate_system_type                                     coordinate_system() const
    {
      return coordinate_system_;
    }
    const std::uniform_real_distribution<result_element_type>& v_distribution   () const
    {
      return v_distribution_;
    }
    const std::uniform_real_distribution<result_element_type>& w_distribution   () const
    {
      return w_distribution_;
    }

  protected:
    result_type                                         mean_             ;
    result_element_type                                 concentration_    ;
    coordinate_system_type                              coordinate_system_;
    std::uniform_real_distribution<result_element_type> v_distribution_   {result_element_type(0), result_element_type(2 * M_PI)};
    std::uniform_real_distribution<result_element_type> w_distribution_   {result_element_type(0), result_element_type(1       )};
  };

  explicit von_mises_fisher_distribution  (const result_type mean, const result_element_type concentration, const coordinate_system_type coordinate_system = coordinate_system_type::cartesian)
  : parameters_(mean, concentration, coordinate_system)
  {

  }
  explicit von_mises_fisher_distribution  (const param_type& parameters)
  : parameters_(parameters)
  {                                        
                                           
  }                                        
  von_mises_fisher_distribution           (const von_mises_fisher_distribution&  that) = default;
  von_mises_fisher_distribution           (      von_mises_fisher_distribution&& temp) = default;
 ~von_mises_fisher_distribution           ()                                           = default;
  von_mises_fisher_distribution& operator=(const von_mises_fisher_distribution&  that) = default;
  von_mises_fisher_distribution& operator=(      von_mises_fisher_distribution&& temp) = default;
  
  void                   reset                ()
  {
    // Intentionally blank.
  }
                         
  param_type             param                () const
  {
    return parameters_;
  }
  void                   param                (const param_type& parameters)
  {
    parameters_ = parameters;
  }
  
  template<typename engine>
  result_type            operator()           (engine& engine) const
  {
    return evaluate(engine, parameters_);
  }
  template<typename engine>
  result_type            operator()           (engine& engine, const param_type& parameters) const
  {
    return evaluate(engine, parameters);
  } 

  result_type            (min)                () const
  {
    return parameters_.coordinate_system() == coordinate_system_type::cartesian 
      ? result_type
      {
        result_element_type(-1), 
        result_element_type(-1), 
        result_element_type(-1)
      }
      : result_type
      {
        result_element_type(1), 
        result_element_type(0), 
        result_element_type(0)
      };
  }
  result_type            (max)                () const
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
        result_element_type(    M_PI)
      };
  }
  
  result_type            mean                 () const
  {
    return parameters_.mean();
  }
  result_element_type    concentration        () const
  {
    return parameters_.concentration();
  }
  coordinate_system_type coordinate_system    () const
  {
    return parameters_.coordinate_system();
  }
  
protected:
  result_element_type    dot_product          (const result_type& lhs, const result_type& rhs) const
  {
    result_element_type dot(0);
    for (auto i = 0; i < 3; ++i)
      dot += lhs[i] * rhs[i];
    return dot;
  }
  result_type            cross_product        (const result_type& lhs, const result_type& rhs) const
  {
    result_type cross;
    cross[0] = lhs[1] * rhs[2] - lhs[2] * rhs[1];
    cross[1] = lhs[2] * rhs[0] - lhs[0] * rhs[2];
    cross[2] = lhs[0] * rhs[1] - lhs[1] * rhs[0];
    return cross;
  }
  
  template<typename engine>
  result_type            evaluate         (engine& engine, const param_type& parameters) const
  {
    auto v  = parameters.v_distribution()(engine);
    auto w  = parameters.w_distribution()(engine);
    auto wi = 1 + std::log(w + (1 - w) * std::exp(-2 * parameters.concentration())) / parameters.concentration();
    auto wd = std::sqrt(1 - std::pow(wi, 2));

    auto result = result_type{std::cos(v) * wd, std::sin(v) * wd, wi};

    if (parameters.mean() != result_type{0, 0, 1})
    {
      auto dot   = dot_product  (result_type{0, 0, 1}, parameters.mean());
      auto cross = cross_product(result_type{0, 0, 1}, parameters.mean());
      auto h     = (1 - dot) / dot_product(cross, cross);
      result = result_type
      {
        dot_product(result_type{ h * cross[0] * cross[0] + dot     , h * cross[0] * cross[1] - cross[2], h * cross[0] * cross[2] + cross[1] }, result),
        dot_product(result_type{ h * cross[0] * cross[1] + cross[2], h * cross[1] * cross[1] + dot     , h * cross[1] * cross[2] - cross[0] }, result),
        dot_product(result_type{ h * cross[0] * cross[2] - cross[1], h * cross[1] * cross[2] + cross[0], h * cross[2] * cross[2] + dot      }, result)
      };
    }
    
    if (parameters.coordinate_system() == coordinate_system_type::spherical)
      result = result_type {1, std::atan2(result[1], result[0]), std::acos(result[2])};

    return result;
  }

  param_type parameters_;
};

template<typename type>
bool operator==(const von_mises_fisher_distribution<type>& lhs, const von_mises_fisher_distribution<type>& rhs)
{
  return lhs.param() == rhs.param();
}
template<typename type>
bool operator!=(const von_mises_fisher_distribution<type>& lhs, const von_mises_fisher_distribution<type>& rhs)
{
  return !(lhs == rhs);
}
template<typename stream_type, typename stream_traits, typename type>
std::basic_ostream<stream_type, stream_traits>& operator<<(std::basic_ostream<stream_type, stream_traits>& stream, const von_mises_fisher_distribution<type>& distribution)
{
  auto mean             = distribution.mean             ();
  auto concentration    = distribution.concentration    ();
  auto coordiate_system = distribution.coordinate_system();

  stream.write(mean.data(), mean.size());
  stream << concentration;
  stream << static_cast<typename std::underlying_type<typename von_mises_fisher_distribution<type>::coordinate_system_type>::type>(coordiate_system);

  return stream;
}
template<typename stream_type, typename stream_traits, typename type>
std::basic_istream<stream_type, stream_traits>& operator>>(std::basic_istream<stream_type, stream_traits>& stream,       von_mises_fisher_distribution<type>& distribution)
{
  typename von_mises_fisher_distribution<type>::result_type                                                 mean             ;
  typename von_mises_fisher_distribution<type>::result_element_type                                         concentration    ;
  typename std::underlying_type<typename von_mises_fisher_distribution<type>::coordinate_system_type>::type coordinate_system;

  stream.read(mean.data(), mean.size());
  stream >> concentration;
  stream >> coordinate_system;

  distribution.param(typename von_mises_fisher_distribution<type>::param_type(mean, concentration, static_cast<typename von_mises_fisher_distribution<type>::coordinate_system_type>(coordinate_system)));

  return stream;
}
}

#endif