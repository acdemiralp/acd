### What is acd?
Single-file utilities for C++, similar in spirit to [github.com/nothings/stb](https://github.com/nothings/stb).

### Documentation

**convert/angles.hpp** 
Degrees to radians and vice versa.

**convert/coordinates.hpp**
Cartesian coordinates to spherical coordinates and vice versa.

**distributions/normal_multivariate_distribution.hpp**
N-dimensional [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution), compliant with `<random>`.

**distributions/uniform_multivariate_distribution.hpp**
N-dimensional multivariate uniform distribution, compliant with `<random>`.

**distributions/uniform_spherical_distribution.hpp**
Uniform distribution on the 2-sphere/hemisphere, compliant with `<random>`.

**distributions/von_mises_fisher_distribution.hpp**
[Von-Mises Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) on the 2-sphere/hemisphere, compliant with `<random>`.

**bitset_enum.hpp**
Declare `struct is_bitset_enum<your_enum> { static const bool enable = true; }` to enable bitset operations on `your_enum`.

**constexpr_for.hpp**
Compile-time for loop as well as its variants for variadic templates and tuples.

**enable_dedicated_gpu.hpp**
Enables AMD and NVIDIA GPUs on laptops or other systems which default to onboard graphics. Include once in main.

**indexing.hpp**
Ravels and unravels N-dimensional indices to 1-dimensional and vice versa, similar to numpy [ravel_multi_index](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ravel_multi_index.html) and [unravel_index](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.unravel_index.html).

**interpolation.hpp**
[Linear interpolation (LERP)](https://en.wikipedia.org/wiki/Linear_interpolation) and [spherical linear interpolation (SLERP)](https://en.wikipedia.org/wiki/Slerp).

**partitioner.hpp**
Partitions an N-dimensional domain to a hyperrectangular grid based on communicator rank and size. Intended for use with MPI.

**permute_for.hpp**
Permutes the loop `for(auto i = start, i < end; i+= step)` over N dimensions.

**prime_factorization.hpp**
Computes the prime factors of the given integer. Useful for partitioning N-dimensional data to a number of threads.

**random_number_generator.hpp**
Encapsulates `<random>` boilerplate. Specify a distribution and go.

**singleton.hpp**
A non-copyable, non-movable singleton.

### Contribution
Additions are very welcome. Just create a merge request to the develop branch.
