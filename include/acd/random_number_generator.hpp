#ifndef ACD_RANDOM_NUMBER_GENERATOR_HPP
#define ACD_RANDOM_NUMBER_GENERATOR_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <random>
#include <utility>

namespace acd
{
template<typename distribution_type = std::uniform_real_distribution<double>>
class random_number_generator
{
public:
  using result_type = typename distribution_type::result_type;

  template<typename... argument_types>
  explicit random_number_generator  (argument_types&&...      arguments   ) 
  : mersenne_twister_(random_device_()), distribution_(std::forward<argument_types>(arguments)...)
  {
    
  }
  explicit random_number_generator  (const distribution_type& distribution) 
  : mersenne_twister_(random_device_()), distribution_(distribution)
  {
    
  }
  random_number_generator           (const random_number_generator&  that) = default;
  random_number_generator           (      random_number_generator&& temp) = default;
  virtual ~random_number_generator  ()                                     = default;
  random_number_generator& operator=(const random_number_generator&  that) = default;
  random_number_generator& operator=(      random_number_generator&& temp) = default;

  result_type                  generate()
  {
    return distribution_(mersenne_twister_);
  }
  template<typename vector_type> 
  vector_type                  generate(std::size_t                size)
  {
    vector_type vector(size);
    std::generate_n(&vector[0], size, function());
    return vector;
  }
  template<typename matrix_type>
  matrix_type                  generate(std::array<std::size_t, 2> size)
  {
    matrix_type matrix(size[0], size[1]);
    std::generate_n(&matrix[0][0], size[0] * size[1], function());
    return matrix;
  }

  std::function<result_type()> function()
  {
    return std::bind(static_cast<result_type(random_number_generator<distribution_type>::*)()>(&random_number_generator<distribution_type>::generate), this);
  }

protected:
  std::random_device random_device_   ;
  std::mt19937       mersenne_twister_;  
  distribution_type  distribution_    ;
};
}

#endif