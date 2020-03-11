#ifndef ACD_GHOST_PARTITIONER_HPP
#define ACD_GHOST_PARTITIONER_HPP

#include "partitioner.hpp"

namespace acd
{
template<std::size_t dimensions>
class ghost_partitioner : public partitioner<dimensions>
{
public:
  using base = partitioner<dimensions>;

  explicit ghost_partitioner(
    const std::size_t                          communicator_rank, 
    const std::size_t                          communicator_size, 
    const std::array<std::size_t, dimensions>& domain_size      ,
    const std::array<std::size_t, dimensions>& ghost_cell_size  )
  : partitioner(communicator_rank, communicator_size, domain_size), ghost_cell_size_(ghost_cell_size)
  {
    ghost_partitioner<dimensions>::update();
  }
  ghost_partitioner           (const ghost_partitioner&  that) = default;
  ghost_partitioner           (      ghost_partitioner&& temp) = default;
  virtual ~ghost_partitioner  ()                               = default;
  ghost_partitioner& operator=(const ghost_partitioner&  that) = default;
  ghost_partitioner& operator=(      ghost_partitioner&& temp) = default;

  void                                       set_ghost_cell_size(const std::array<std::size_t, dimensions>& ghost_cell_size)
  {
    ghost_cell_size_ = ghost_cell_size;
    ghost_partitioner<dimensions>::update();
  }

  const std::array<std::size_t, dimensions>& ghost_cell_size    () const
  {
    return ghost_cell_size_;
  }
  const std::array<std::size_t, dimensions>& ghosted_block_size () const
  {
    return ghosted_block_size_;
  }
  const std::array<std::size_t, dimensions>& ghosted_rank_offset() const
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

  std::array<std::size_t, dimensions> ghost_cell_size_    ;
  std::array<std::size_t, dimensions> ghosted_block_size_ ;
  std::array<std::size_t, dimensions> ghosted_rank_offset_;
};
}

#endif