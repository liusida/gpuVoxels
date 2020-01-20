#if !defined(TI_UTILS_H)
#define TI_UTILS_H
#include <stdexcept>

#include "VX3.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE
#endif

#include "types.h"
#include "const.h"
// #include "TI_vector.h"
#include "TI_Vec3D.h"
#include "TI_Quat3D.h"

#endif // TI_UTILS_H
