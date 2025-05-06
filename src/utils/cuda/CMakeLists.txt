set(CUDA_UTILS_SRC
    error.h
    error.cpp
    device.h
    device.cpp
    event.h
    memory.h
    array.h
)

add_library(cuda_utils STATIC ${CUDA_UTILS_SRC})
set_target_properties(cuda_utils PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED ON)
target_include_directories(cuda_utils PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
