add_subdirectory(utils)
add_subdirectory(tools)

set(HDR_PIPELINE_SRC
    envmap.cpp
    envmap.h
    hdr_pipeline.h
    hdr_pipeline.cpp
    hdr_pipeline.cu
    main.cpp
)
add_executable(hdr_pipeline ${HDR_PIPELINE_SRC})
set_target_properties(hdr_pipeline PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(hdr_pipeline utils cuda_utils)

# 添加CUDA编译选项以支持性能分析
target_compile_options(hdr_pipeline PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
