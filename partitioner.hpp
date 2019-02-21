#ifndef ACD_PARTITIONER_HPP
#define ACD_PARTITIONER_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>

#include "indexing.hpp"
#include "prime_factorization.hpp"

namespace acd
{
// Partitions an N-dimensional domain of domain_size to a hyperrectangular grid of grid_size, 
// which consists of blocks of block_size based on the communicator_size (process/thread count).
// Also computes a rank_multi_index which is the N-dimensional index of the rank, and a 
// rank_offset which equals block_size * rank_multi_index, based on the communicator_rank 
// (process/thread index). Intended for use with MPI.
template<std::size_t dimensions>
class partitioner
{
public:
  explicit partitioner(
    const std::size_t                          communicator_rank, 
    const std::size_t                          communicator_size, 
    const std::array<std::size_t, dimensions>& domain_size      )
  : communicator_rank_(communicator_rank)
  , communicator_size_(communicator_size)
  , domain_size_      (domain_size      )
  {
    update();
  }
  partitioner           (const partitioner&  that) = default;
  partitioner           (      partitioner&& temp) = default;
 ~partitioner           ()                         = default;
  partitioner& operator=(const partitioner&  that) = default;
  partitioner& operator=(      partitioner&& temp) = default;

  void set_communicator_rank(const std::size_t                          communicator_rank)
  {
    communicator_rank_ = communicator_rank;
    update();
  }
  void set_communicator_size(const std::size_t                          communicator_size)
  {
    communicator_size_ = communicator_size;
    update();
  }
  void set_domain_size      (const std::array<std::size_t, dimensions>& domain_size      )
  {
    domain_size_ = domain_size;
    update();
  }
  
  const std::size_t                          communicator_rank() const
  {
    return communicator_rank_;
  }
  const std::size_t                          communicator_size() const
  {
    return communicator_size_;
  }
  const std::array<std::size_t, dimensions>& domain_size      () const
  {
    return domain_size_;
  }

  const std::array<std::size_t, dimensions>& grid_size        () const
  {
    return grid_size_;
  }
  const std::array<std::size_t, dimensions>& block_size       () const
  {
    return block_size_;
  }
  const std::array<std::size_t, dimensions>& rank_multi_index () const
  {
    return rank_multi_index_;
  }
  const std::array<std::size_t, dimensions>& rank_offset      () const
  {
    return rank_offset_;
  }

protected:
  void update()
  {
    auto prime_factors = prime_factorize(communicator_size_);
    auto current_size  = domain_size_;
    grid_size_.fill(1);
    while (!prime_factors.empty())
    {
      auto dimension = std::distance(current_size.begin(), std::max_element(current_size.begin(), current_size.end()));
      current_size[dimension] /= prime_factors.back();
      grid_size_  [dimension] *= prime_factors.back();
      prime_factors.pop_back();
    }
    std::transform(domain_size_.begin(), domain_size_.end(), grid_size_       .begin(), block_size_ .begin(), std::divides   <>());
    rank_multi_index_ = unravel_index<dimensions>(communicator_rank_, grid_size_);
    std::transform(block_size_ .begin(), block_size_ .end(), rank_multi_index_.begin(), rank_offset_.begin(), std::multiplies<>());
  }
  
  std::size_t                         communicator_rank_;
  std::size_t                         communicator_size_;
  std::array<std::size_t, dimensions> domain_size_      ;

  std::array<std::size_t, dimensions> grid_size_        ;
  std::array<std::size_t, dimensions> block_size_       ;
  std::array<std::size_t, dimensions> rank_multi_index_ ;
  std::array<std::size_t, dimensions> rank_offset_      ;
};
}

#endif
