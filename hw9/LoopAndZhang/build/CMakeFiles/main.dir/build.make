# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

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
CMAKE_COMMAND = /opt/homebrew/lib/python3.11/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/lib/python3.11/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/build"

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/main.cpp.o: /Users/michaelg/Desktop/ECE\ 661/hw9/LoopAndZhang/src/main.cpp
CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/src/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/main.cpp.o -MF CMakeFiles/main.dir/src/main.cpp.o.d -o CMakeFiles/main.dir/src/main.cpp.o -c "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/src/main.cpp"

CMakeFiles/main.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/src/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/src/main.cpp" > CMakeFiles/main.dir/src/main.cpp.i

CMakeFiles/main.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/src/main.cpp" -o CMakeFiles/main.dir/src/main.cpp.s

CMakeFiles/main.dir/src/util.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/util.cpp.o: /Users/michaelg/Desktop/ECE\ 661/hw9/LoopAndZhang/src/util.cpp
CMakeFiles/main.dir/src/util.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/src/util.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/util.cpp.o -MF CMakeFiles/main.dir/src/util.cpp.o.d -o CMakeFiles/main.dir/src/util.cpp.o -c "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/src/util.cpp"

CMakeFiles/main.dir/src/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/src/util.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/src/util.cpp" > CMakeFiles/main.dir/src/util.cpp.i

CMakeFiles/main.dir/src/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/util.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/src/util.cpp" -o CMakeFiles/main.dir/src/util.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/src/main.cpp.o" \
"CMakeFiles/main.dir/src/util.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/src/main.cpp.o
main: CMakeFiles/main.dir/src/util.cpp.o
main: CMakeFiles/main.dir/build.make
main: /opt/miniconda3/envs/ece661/lib/libopencv_gapi.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_stitching.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_alphamat.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_aruco.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_bgsegm.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_bioinspired.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_ccalib.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_cvv.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_dnn_objdetect.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_dnn_superres.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_dpm.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_face.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_freetype.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_fuzzy.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_hdf.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_hfs.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_img_hash.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_intensity_transform.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_line_descriptor.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_mcc.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_quality.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_rapid.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_reg.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_rgbd.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_saliency.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_signal.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_stereo.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_structured_light.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_superres.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_surface_matching.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_tracking.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_videostab.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_wechat_qrcode.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_xfeatures2d.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_xobjdetect.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_xphoto.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_shape.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_highgui.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_datasets.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_plot.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_text.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_ml.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_phase_unwrapping.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_optflow.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_ximgproc.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_video.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_videoio.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_imgcodecs.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_objdetect.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_calib3d.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_dnn.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_features2d.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_flann.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_photo.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_imgproc.4.10.0.dylib
main: /opt/miniconda3/envs/ece661/lib/libopencv_core.4.10.0.dylib
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang" "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang" "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/build" "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/build" "/Users/michaelg/Desktop/ECE 661/hw9/LoopAndZhang/build/CMakeFiles/main.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : CMakeFiles/main.dir/depend

