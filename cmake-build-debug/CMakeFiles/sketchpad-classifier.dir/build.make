# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\sketchpad-classifier.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\sketchpad-classifier.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\sketchpad-classifier.dir\flags.make

CMakeFiles\sketchpad-classifier.dir\apps\cinder_app_main.cc.obj: CMakeFiles\sketchpad-classifier.dir\flags.make
CMakeFiles\sketchpad-classifier.dir\apps\cinder_app_main.cc.obj: ..\apps\cinder_app_main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sketchpad-classifier.dir/apps/cinder_app_main.cc.obj"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\sketchpad-classifier.dir\apps\cinder_app_main.cc.obj /FdCMakeFiles\sketchpad-classifier.dir\ /FS -c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\apps\cinder_app_main.cc
<<

CMakeFiles\sketchpad-classifier.dir\apps\cinder_app_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sketchpad-classifier.dir/apps/cinder_app_main.cc.i"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe > CMakeFiles\sketchpad-classifier.dir\apps\cinder_app_main.cc.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\apps\cinder_app_main.cc
<<

CMakeFiles\sketchpad-classifier.dir\apps\cinder_app_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sketchpad-classifier.dir/apps/cinder_app_main.cc.s"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\sketchpad-classifier.dir\apps\cinder_app_main.cc.s /c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\apps\cinder_app_main.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\core\image.cc.obj: CMakeFiles\sketchpad-classifier.dir\flags.make
CMakeFiles\sketchpad-classifier.dir\src\core\image.cc.obj: ..\src\core\image.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sketchpad-classifier.dir/src/core/image.cc.obj"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\sketchpad-classifier.dir\src\core\image.cc.obj /FdCMakeFiles\sketchpad-classifier.dir\ /FS -c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\core\image.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\core\image.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sketchpad-classifier.dir/src/core/image.cc.i"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe > CMakeFiles\sketchpad-classifier.dir\src\core\image.cc.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\core\image.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\core\image.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sketchpad-classifier.dir/src/core/image.cc.s"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\sketchpad-classifier.dir\src\core\image.cc.s /c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\core\image.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\core\model.cc.obj: CMakeFiles\sketchpad-classifier.dir\flags.make
CMakeFiles\sketchpad-classifier.dir\src\core\model.cc.obj: ..\src\core\model.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/sketchpad-classifier.dir/src/core/model.cc.obj"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\sketchpad-classifier.dir\src\core\model.cc.obj /FdCMakeFiles\sketchpad-classifier.dir\ /FS -c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\core\model.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\core\model.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sketchpad-classifier.dir/src/core/model.cc.i"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe > CMakeFiles\sketchpad-classifier.dir\src\core\model.cc.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\core\model.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\core\model.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sketchpad-classifier.dir/src/core/model.cc.s"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\sketchpad-classifier.dir\src\core\model.cc.s /c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\core\model.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\core\dataset.cc.obj: CMakeFiles\sketchpad-classifier.dir\flags.make
CMakeFiles\sketchpad-classifier.dir\src\core\dataset.cc.obj: ..\src\core\dataset.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/sketchpad-classifier.dir/src/core/dataset.cc.obj"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\sketchpad-classifier.dir\src\core\dataset.cc.obj /FdCMakeFiles\sketchpad-classifier.dir\ /FS -c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\core\dataset.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\core\dataset.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sketchpad-classifier.dir/src/core/dataset.cc.i"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe > CMakeFiles\sketchpad-classifier.dir\src\core\dataset.cc.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\core\dataset.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\core\dataset.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sketchpad-classifier.dir/src/core/dataset.cc.s"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\sketchpad-classifier.dir\src\core\dataset.cc.s /c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\core\dataset.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\utility\train_utility.cc.obj: CMakeFiles\sketchpad-classifier.dir\flags.make
CMakeFiles\sketchpad-classifier.dir\src\utility\train_utility.cc.obj: ..\src\utility\train_utility.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/sketchpad-classifier.dir/src/utility/train_utility.cc.obj"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\sketchpad-classifier.dir\src\utility\train_utility.cc.obj /FdCMakeFiles\sketchpad-classifier.dir\ /FS -c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\utility\train_utility.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\utility\train_utility.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sketchpad-classifier.dir/src/utility/train_utility.cc.i"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe > CMakeFiles\sketchpad-classifier.dir\src\utility\train_utility.cc.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\utility\train_utility.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\utility\train_utility.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sketchpad-classifier.dir/src/utility/train_utility.cc.s"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\sketchpad-classifier.dir\src\utility\train_utility.cc.s /c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\utility\train_utility.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\visualizer\naive_bayes_app.cc.obj: CMakeFiles\sketchpad-classifier.dir\flags.make
CMakeFiles\sketchpad-classifier.dir\src\visualizer\naive_bayes_app.cc.obj: ..\src\visualizer\naive_bayes_app.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/sketchpad-classifier.dir/src/visualizer/naive_bayes_app.cc.obj"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\sketchpad-classifier.dir\src\visualizer\naive_bayes_app.cc.obj /FdCMakeFiles\sketchpad-classifier.dir\ /FS -c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\visualizer\naive_bayes_app.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\visualizer\naive_bayes_app.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sketchpad-classifier.dir/src/visualizer/naive_bayes_app.cc.i"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe > CMakeFiles\sketchpad-classifier.dir\src\visualizer\naive_bayes_app.cc.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\visualizer\naive_bayes_app.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\visualizer\naive_bayes_app.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sketchpad-classifier.dir/src/visualizer/naive_bayes_app.cc.s"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\sketchpad-classifier.dir\src\visualizer\naive_bayes_app.cc.s /c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\visualizer\naive_bayes_app.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\visualizer\sketchpad.cc.obj: CMakeFiles\sketchpad-classifier.dir\flags.make
CMakeFiles\sketchpad-classifier.dir\src\visualizer\sketchpad.cc.obj: ..\src\visualizer\sketchpad.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/sketchpad-classifier.dir/src/visualizer/sketchpad.cc.obj"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\sketchpad-classifier.dir\src\visualizer\sketchpad.cc.obj /FdCMakeFiles\sketchpad-classifier.dir\ /FS -c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\visualizer\sketchpad.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\visualizer\sketchpad.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sketchpad-classifier.dir/src/visualizer/sketchpad.cc.i"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe > CMakeFiles\sketchpad-classifier.dir\src\visualizer\sketchpad.cc.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\visualizer\sketchpad.cc
<<

CMakeFiles\sketchpad-classifier.dir\src\visualizer\sketchpad.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sketchpad-classifier.dir/src/visualizer/sketchpad.cc.s"
	C:\PROGRA~2\MICROS~1.0\VC\bin\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\sketchpad-classifier.dir\src\visualizer\sketchpad.cc.s /c C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\src\visualizer\sketchpad.cc
<<

# Object files for target sketchpad-classifier
sketchpad__classifier_OBJECTS = \
"CMakeFiles\sketchpad-classifier.dir\apps\cinder_app_main.cc.obj" \
"CMakeFiles\sketchpad-classifier.dir\src\core\image.cc.obj" \
"CMakeFiles\sketchpad-classifier.dir\src\core\model.cc.obj" \
"CMakeFiles\sketchpad-classifier.dir\src\core\dataset.cc.obj" \
"CMakeFiles\sketchpad-classifier.dir\src\utility\train_utility.cc.obj" \
"CMakeFiles\sketchpad-classifier.dir\src\visualizer\naive_bayes_app.cc.obj" \
"CMakeFiles\sketchpad-classifier.dir\src\visualizer\sketchpad.cc.obj"

# External object files for target sketchpad-classifier
sketchpad__classifier_EXTERNAL_OBJECTS =

Debug\sketchpad-classifier\sketchpad-classifier.exe: CMakeFiles\sketchpad-classifier.dir\apps\cinder_app_main.cc.obj
Debug\sketchpad-classifier\sketchpad-classifier.exe: CMakeFiles\sketchpad-classifier.dir\src\core\image.cc.obj
Debug\sketchpad-classifier\sketchpad-classifier.exe: CMakeFiles\sketchpad-classifier.dir\src\core\model.cc.obj
Debug\sketchpad-classifier\sketchpad-classifier.exe: CMakeFiles\sketchpad-classifier.dir\src\core\dataset.cc.obj
Debug\sketchpad-classifier\sketchpad-classifier.exe: CMakeFiles\sketchpad-classifier.dir\src\utility\train_utility.cc.obj
Debug\sketchpad-classifier\sketchpad-classifier.exe: CMakeFiles\sketchpad-classifier.dir\src\visualizer\naive_bayes_app.cc.obj
Debug\sketchpad-classifier\sketchpad-classifier.exe: CMakeFiles\sketchpad-classifier.dir\src\visualizer\sketchpad.cc.obj
Debug\sketchpad-classifier\sketchpad-classifier.exe: CMakeFiles\sketchpad-classifier.dir\build.make
Debug\sketchpad-classifier\sketchpad-classifier.exe: C:\Users\amaan\CppLibraries\Cinder\lib\msw\x86\Debug\v140\cinder.lib
Debug\sketchpad-classifier\sketchpad-classifier.exe: CMakeFiles\sketchpad-classifier.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable Debug\sketchpad-classifier\sketchpad-classifier.exe"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E vs_link_exe --intdir=CMakeFiles\sketchpad-classifier.dir --rc=C:\PROGRA~2\WI3CF2~1\8.1\bin\x86\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\8.1\bin\x86\mt.exe --manifests  -- C:\PROGRA~2\MICROS~1.0\VC\bin\link.exe /nologo @CMakeFiles\sketchpad-classifier.dir\objects1.rsp @<<
 /out:Debug\sketchpad-classifier\sketchpad-classifier.exe /implib:sketchpad-classifier.lib /pdb:C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug\Debug\sketchpad-classifier\sketchpad-classifier.pdb /version:0.0  /machine:X86 /debug /INCREMENTAL /subsystem:windows /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:LIBCPMT   -LIBPATH:C:\Users\amaan\CppLibraries\Cinder\lib\msw\x86  C:\Users\amaan\CppLibraries\Cinder\lib\msw\x86\Debug\v140\cinder.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<

# Rule to build all files generated by this target.
CMakeFiles\sketchpad-classifier.dir\build: Debug\sketchpad-classifier\sketchpad-classifier.exe

.PHONY : CMakeFiles\sketchpad-classifier.dir\build

CMakeFiles\sketchpad-classifier.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\sketchpad-classifier.dir\cmake_clean.cmake
.PHONY : CMakeFiles\sketchpad-classifier.dir\clean

CMakeFiles\sketchpad-classifier.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\cmake-build-debug\CMakeFiles\sketchpad-classifier.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\sketchpad-classifier.dir\depend

