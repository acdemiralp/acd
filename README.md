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

**binomial_coefficient.hpp**
Computes binomial coefficients at compile-time and runtime.

**bitset_enum.hpp**
Declare `struct is_bitset_enum<your_enum> { static const bool enable = true; }` to enable bitset operations on `your_enum`.

**clamp.hpp**
Constrains a value to be within a specified range (min, max).

**constexpr_for.hpp**
Compile-time for loop as well as its variants for variadic templates and tuples.

**enable_dedicated_gpu.hpp**
Enables AMD and NVIDIA GPUs on laptops or other systems which default to onboard graphics. Include once in main.

**factorial.hpp**
Computes factorial at compile-time and runtime.

**gcd.hpp**
Computes the [greatest common divisor (GCD)](https://en.wikipedia.org/wiki/Greatest_common_divisor) of two integers.

**indexing.hpp**
Ravels and unravels N-dimensional indices to 1-dimensional and vice versa, similar to numpy [ravel_multi_index](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ravel_multi_index.html) and [unravel_index](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.unravel_index.html).

**interpolation.hpp**
[Linear interpolation (LERP)](https://en.wikipedia.org/wiki/Linear_interpolation) and [spherical linear interpolation (SLERP)](https://en.wikipedia.org/wiki/Slerp).

**lcm.hpp**
Computes the [least common multiple (LCM)](https://en.wikipedia.org/wiki/Least_common_multiple) of two integers.

**map_range.hpp**
Maps a value from one range to another range (similar to Arduino's map function).

**partitioner.hpp**
Partitions an N-dimensional domain to a hyperrectangular grid based on communicator rank and size. Intended for use with MPI.

**permute_for.hpp**
Permutes the loop `for(auto i = start, i < end; i+= step)` over N dimensions.

**power_of_two.hpp**
Checks if a number is a power of two, and finds the next or previous power of two.

**prime_factorization.hpp**
Computes the prime factors of the given integer. Useful for partitioning N-dimensional data to a number of threads.

**random_number_generator.hpp**
Encapsulates `<random>` boilerplate. Specify a distribution and go.

**sign.hpp**
Returns the sign of a number (-1, 0, or 1).

**singleton.hpp**
A non-copyable, non-movable singleton.

### Contribution
Additions are very welcome. Just create a merge request to the develop branch.
