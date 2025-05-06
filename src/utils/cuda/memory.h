#ifndef INCLUDED_UTILS_CUDA_MEMORY
#define INCLUDED_UTILS_CUDA_MEMORY

#include <cstddef>
#include <memory>

#include <cuda_runtime_api.h>

#include "error.h"

namespace CUDA {
    struct FreeDeleter {
        void operator()(void* ptr) const {
            cudaFree(ptr);
        }
    };

    template <typename T>
    using unique_ptr = std::unique_ptr<T, FreeDeleter>;

    template <typename T>
    auto malloc(std::size_t size) {
        void* ptr;
        throw_error(cudaMalloc(&ptr, size * sizeof(T)));
        return unique_ptr<T> { static_cast<T*>(ptr) };
    }

    template <typename T>
    auto malloc_zeroed(std::size_t size) {
        auto memory = malloc<T>(size);
        throw_error(cudaMemset(memory.get(), 0, size * sizeof(T)));
        return memory;
    }
}

#endif // INCLUDED_UTILS_CUDA_MEMORY
