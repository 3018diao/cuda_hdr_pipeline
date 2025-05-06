#ifndef INCLUDED_UTILS_CUDA_DEVICE
#define INCLUDED_UTILS_CUDA_DEVICE

#include <iostream>

#include <cuda_runtime_api.h>

namespace CUDA {
    std::ostream& print_device_properties(std::ostream& out, int device);
}

#endif // INCLUDED_UTILS_CUDA_DEVICE
