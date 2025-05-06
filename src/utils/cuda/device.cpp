#include <iostream>
#include <iomanip>

#include "error.h"
#include "device.h"

namespace CUDA {
    std::ostream& print_device_properties(std::ostream& out, int device) {
        cudaDeviceProp props;
        throw_error(cudaGetDeviceProperties(&props, device));
        return out << "using cuda device " << device << ":\n"
                      "\t" << props.name << "\n"
                      "\tcompute capability " << props.major << "." << props.minor << " @ " << std::setprecision(1) << std::fixed << props.clockRate / 1000.0f << " MHz\n"
                      "\t" << props.multiProcessorCount << " multiprocessors\n"
                      "\t" << props.totalGlobalMem / (1024U * 1024U) << " MiB global memory  " << props.sharedMemPerMultiprocessor / 1024 << " KiB shared memory";
    }
}
