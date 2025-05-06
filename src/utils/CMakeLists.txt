add_subdirectory(cuda)

find_package(PNG REQUIRED)

set(UTILS_SRC
    io.h
    io/image_io.h
    io/radiance.h
    io/radiance.cpp
    io/pfm.h
    io/pfm.cpp
    io/png.h
    io/png.cpp
    io/obj_stream.h
    io/obj_reader.h
    io/obj.h
    io/obj.cpp
    argparse.cpp
    argparse.h
)
add_library(utils STATIC ${UTILS_SRC})
set_target_properties(utils PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_link_libraries(utils PRIVATE PNG::PNG)
