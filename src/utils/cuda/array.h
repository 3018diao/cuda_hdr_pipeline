#ifndef INCLUDED_UTILS_CUDA_ARRAY
#define INCLUDED_UTILS_CUDA_ARRAY

#include <cstddef>
#include <memory>

#include <cuda_runtime_api.h>

#include "error.h"

namespace CUDA {
    struct FreeArrayDeleter {
        void operator()(cudaArray_t arr) const {
            cudaFreeArray(arr);
        }
    };

    using unique_array = std::unique_ptr<cudaArray, FreeArrayDeleter>;

    inline unique_array create_array(std::size_t width, std::size_t height, const cudaChannelFormatDesc& channel_desc, unsigned int flags = 0U) {
        cudaArray_t arr;
        throw_error(cudaMallocArray(&arr, &channel_desc, width, height, flags));
        return unique_array(arr);
    }
}

#endif // INCLUDED_UTILS_CUDA_ARRAY
