#pragma once

#include "partitioner.hpp"

namespace acd
{
template<std::size_t dimensions, typename size_type = std::size_t, typename container_type = std::array<size_type, dimensions>>
class ghost_partitioner : public partitioner<dimensions, size_type, container_type>
{
public:
  using base = partitioner<dimensions>;

  explicit ghost_partitioner(
    const size_type       communicator_rank, 
    const size_type       communicator_size, 
    const container_type& domain_size      ,
    const container_type& ghost_cell_size  )
  : partitioner(communicator_rank, communicator_size, domain_size), ghost_cell_size_(ghost_cell_size)
  {
    ghost_partitioner<dimensions>::update();
  }
  ghost_partitioner           (const ghost_partitioner&  that) = default;
  ghost_partitioner           (      ghost_partitioner&& temp) = default;
  virtual ~ghost_partitioner  ()                               = default;
  ghost_partitioner& operator=(const ghost_partitioner&  that) = default;
  ghost_partitioner& operator=(      ghost_partitioner&& temp) = default;

  void                                       set_ghost_cell_size(const container_type& ghost_cell_size)
  {
    ghost_cell_size_ = ghost_cell_size;
    ghost_partitioner<dimensions>::update();
  }

  const container_type& ghost_cell_size    () const
  {
    return ghost_cell_size_;
  }
  const container_type& ghosted_block_size () const
  {
    return ghosted_block_size_;
  }
  const container_type& ghosted_rank_offset() const
  {
    return ghosted_rank_offset_;
  }

protected:
  void update() override
  {
    base::update();
    for (auto i = 0; i < dimensions; ++i)
    {
      if (base::rank_offset_[i] + base::block_size_ [i] + ghost_cell_size_  [i] < base::domain_size_[i])
        ghosted_block_size_ [i] = base::block_size_ [i] + ghost_cell_size_  [i];
      else
        ghosted_block_size_ [i] = base::domain_size_[i] - base::rank_offset_[i];

      if (base::rank_offset_[i] >= ghost_cell_size_ [i])
        ghosted_rank_offset_[i] = base::rank_offset_[i] - ghost_cell_size_  [i];
      else
        ghosted_rank_offset_[i] = 0;
    }
  }

  container_type ghost_cell_size_    ;
  container_type ghosted_block_size_ ;
  container_type ghosted_rank_offset_;
};
}