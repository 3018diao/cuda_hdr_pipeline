# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/huiyu/hdr-pipeline-task1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/huiyu/hdr-pipeline-task1/build

# Include any dependencies generated for this target.
include src/tools/imgdiff/CMakeFiles/imgdiff.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/tools/imgdiff/CMakeFiles/imgdiff.dir/compiler_depend.make

# Include the progress variables for this target.
include src/tools/imgdiff/CMakeFiles/imgdiff.dir/progress.make

# Include the compile flags for this target's objects.
include src/tools/imgdiff/CMakeFiles/imgdiff.dir/flags.make

src/tools/imgdiff/CMakeFiles/imgdiff.dir/main.cpp.o: src/tools/imgdiff/CMakeFiles/imgdiff.dir/flags.make
src/tools/imgdiff/CMakeFiles/imgdiff.dir/main.cpp.o: ../src/tools/imgdiff/main.cpp
src/tools/imgdiff/CMakeFiles/imgdiff.dir/main.cpp.o: src/tools/imgdiff/CMakeFiles/imgdiff.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/huiyu/hdr-pipeline-task1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/tools/imgdiff/CMakeFiles/imgdiff.dir/main.cpp.o"
	cd /home/huiyu/hdr-pipeline-task1/build/src/tools/imgdiff && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/tools/imgdiff/CMakeFiles/imgdiff.dir/main.cpp.o -MF CMakeFiles/imgdiff.dir/main.cpp.o.d -o CMakeFiles/imgdiff.dir/main.cpp.o -c /home/huiyu/hdr-pipeline-task1/src/tools/imgdiff/main.cpp

src/tools/imgdiff/CMakeFiles/imgdiff.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/imgdiff.dir/main.cpp.i"
	cd /home/huiyu/hdr-pipeline-task1/build/src/tools/imgdiff && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/huiyu/hdr-pipeline-task1/src/tools/imgdiff/main.cpp > CMakeFiles/imgdiff.dir/main.cpp.i

src/tools/imgdiff/CMakeFiles/imgdiff.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/imgdiff.dir/main.cpp.s"
	cd /home/huiyu/hdr-pipeline-task1/build/src/tools/imgdiff && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/huiyu/hdr-pipeline-task1/src/tools/imgdiff/main.cpp -o CMakeFiles/imgdiff.dir/main.cpp.s

# Object files for target imgdiff
imgdiff_OBJECTS = \
"CMakeFiles/imgdiff.dir/main.cpp.o"

# External object files for target imgdiff
imgdiff_EXTERNAL_OBJECTS =

bin/imgdiff: src/tools/imgdiff/CMakeFiles/imgdiff.dir/main.cpp.o
bin/imgdiff: src/tools/imgdiff/CMakeFiles/imgdiff.dir/build.make
bin/imgdiff: lib/libutils.a
bin/imgdiff: /usr/lib/x86_64-linux-gnu/libpng.so
bin/imgdiff: /usr/lib/x86_64-linux-gnu/libz.so
bin/imgdiff: src/tools/imgdiff/CMakeFiles/imgdiff.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/huiyu/hdr-pipeline-task1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/imgdiff"
	cd /home/huiyu/hdr-pipeline-task1/build/src/tools/imgdiff && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imgdiff.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/tools/imgdiff/CMakeFiles/imgdiff.dir/build: bin/imgdiff
.PHONY : src/tools/imgdiff/CMakeFiles/imgdiff.dir/build

src/tools/imgdiff/CMakeFiles/imgdiff.dir/clean:
	cd /home/huiyu/hdr-pipeline-task1/build/src/tools/imgdiff && $(CMAKE_COMMAND) -P CMakeFiles/imgdiff.dir/cmake_clean.cmake
.PHONY : src/tools/imgdiff/CMakeFiles/imgdiff.dir/clean

src/tools/imgdiff/CMakeFiles/imgdiff.dir/depend:
	cd /home/huiyu/hdr-pipeline-task1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/huiyu/hdr-pipeline-task1 /home/huiyu/hdr-pipeline-task1/src/tools/imgdiff /home/huiyu/hdr-pipeline-task1/build /home/huiyu/hdr-pipeline-task1/build/src/tools/imgdiff /home/huiyu/hdr-pipeline-task1/build/src/tools/imgdiff/CMakeFiles/imgdiff.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/tools/imgdiff/CMakeFiles/imgdiff.dir/depend

