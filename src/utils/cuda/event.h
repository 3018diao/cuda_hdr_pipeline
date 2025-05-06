#ifndef INCLUDED_UTILS_CUDA_EVENT
#define INCLUDED_UTILS_CUDA_EVENT

#include <type_traits>
#include <memory>

#include <cuda_runtime_api.h>

#include "error.h"

namespace CUDA {
    struct DestroyEventDeleter {
        void operator()(cudaEvent_t event) const {
            cudaEventDestroy(event);
        }
    };

    using unique_event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, DestroyEventDeleter>;

    inline unique_event create_event() {
        cudaEvent_t event;
        throw_error(cudaEventCreate(&event));
        return unique_event(event);
    }

    inline unique_event create_event(unsigned int flags) {
        cudaEvent_t event;
        throw_error(cudaEventCreateWithFlags(&event, flags));
        return unique_event(event);
    }

    inline float elapsed_time(cudaEvent_t start, cudaEvent_t end) {
        float t;
        throw_error(cudaEventElapsedTime(&t, start, end));
        return t;
    }
}

#endif // INCLUDED_UTILS_CUDA_EVENT
