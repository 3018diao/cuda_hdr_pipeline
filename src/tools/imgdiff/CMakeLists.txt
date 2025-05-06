add_executable(imgdiff main.cpp)
set_target_properties(imgdiff PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_link_libraries(imgdiff utils)
