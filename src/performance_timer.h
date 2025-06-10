#ifndef INCLUDED_PERFORMANCE_TIMER
#define INCLUDED_PERFORMANCE_TIMER

#include <cuda_runtime_api.h>
#include <iostream>

class PerformanceTimer {
    cudaEvent_t start, stop;
    const char* name;
    
public:
    PerformanceTimer(const char* timer_name) : name(timer_name) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    
    ~PerformanceTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "[TIMER] " << name << ": " << ms << " ms" << std::endl;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

// 便利宏
#define CUDA_TIMER(name) PerformanceTimer timer(name)

#endif // INCLUDED_PERFORMANCE_TIMER 