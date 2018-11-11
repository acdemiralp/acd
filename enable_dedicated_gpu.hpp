#ifndef ACD_ENABLE_DEDICATED_GPU_HPP
#define ACD_ENABLE_DEDICATED_GPU_HPP

#include <cstdint>

#if _WIN32 || _WIN64
extern "C"
{
inline __declspec(dllexport) std::int32_t  AmdPowerXpressRequestHighPerformance = 1;
inline __declspec(dllexport) std::uint32_t NvOptimusEnablement                  = 0x00000001;
}
#elif
{
inline                       std::int32_t  AmdPowerXpressRequestHighPerformance = 1;
inline                       std::uint32_t NvOptimusEnablement                  = 0x00000001;
}
#endif

#endif