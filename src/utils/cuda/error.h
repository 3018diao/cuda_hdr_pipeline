#ifndef INCLUDED_UTILS_CUDA_ERROR
#define INCLUDED_UTILS_CUDA_ERROR

#include <exception>

#include <cuda_runtime_api.h>

namespace CUDA {
    class error : public std::exception {
        cudaError err;

    public:
        error(cudaError err);

        const char* what() const noexcept override;
    };

    inline void throw_error(cudaError err) {
        if (err != cudaSuccess)
            throw error(err);
    }
}

using CUDA::throw_error;

#endif // INCLUDED_CUDA_ERROR
