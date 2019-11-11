# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
cmake_minimum_required(VERSION 3.0)

include(CheckFunctionExists)
include(CheckCXXSourceRuns)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  message(FATAL_ERROR "The build directory must be different from the \
        root directory of this software.")
endif()

cmake_policy(SET CMP0048 NEW)
project(pyinterp LANGUAGES CXX)

if (POLICY CMP0063)
  cmake_policy(SET CMP0063 NEW)
endif ()

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif ()

if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif ()

# CMake module search path
set(
  CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/tools;"
  "${CMAKE_MODULE_PATH}"
)

# By default, build type is set to release, with debugging information.
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
message("-- Build type: ${CMAKE_BUILD_TYPE}")

# The library must be built using C++17 compiler.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MACOSX_RPATH 1)

include(CheckCXXCompilerFlag)
if(NOT WIN32)
  check_cxx_compiler_flag("-std=c++17" HAS_CPP17_FLAG)
else()
  check_cxx_compiler_flag("/std:c++17" HAS_CPP17_FLAG)
endif()
if(NOT HAS_CPP17_FLAG)
  message(FATAL_ERROR "Unsupported compiler -- requires C++17 support!")
endif()

macro(check_cxx_compiler_and_linker_flags _RESULT _CXX_FLAGS _LINKER_FLAGS)
  set(CMAKE_REQUIRED_FLAGS ${_CXX_FLAGS})
  set(CMAKE_REQUIRED_LIBRARIES ${_LINKER_FLAGS})
  set(CMAKE_REQUIRED_QUIET FALSE)
  check_cxx_source_runs("int main(int argc, char **argv) { return 0; }" ${_RESULT})
  set(CMAKE_REQUIRED_FLAGS "")
  set(CMAKE_REQUIRED_LIBRARIES "")
  unset(_RESULT)
endmacro()

# Always use libc++ on Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  check_cxx_compiler_and_linker_flags(
    HAS_LIBCPP "-stdlib=libc++" "-stdlib=libc++")
  if (HAS_LIBCPP)
    string(APPEND CMAKE_CXX_FLAGS " -stdlib=libc++")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -stdlib=libc++")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -stdlib=libc++")
    check_cxx_compiler_and_linker_flags(
      HAS_LIBCPPABI "-stdlib=libc++" "-stdlib=libc++ -lc++abi")
    if(HAS_LIBCPPABI)
      string(APPEND CMAKE_EXE_LINKER_FLAGS " -lc++abi")
      string(APPEND CMAKE_SHARED_LINKER_FLAGS " -lc++abi")
    endif()
  endif()
  check_cxx_compiler_and_linker_flags(
    HAS_SIZED_DEALLOCATION "-fsized-deallocation" "")
  if(HAS_SIZED_DEALLOCATION)
    string(APPEND CMAKE_CXX_FLAGS " -fsized-deallocation")
  endif()
endif()

if(NOT WIN32)
  if(NOT CMAKE_CXX_FLAGS MATCHES "-Wall$")
    string(APPEND CMAKE_CXX_FLAGS " -Wall")
  endif()
  if(NOT CMAKE_CXX_COMPILER MATCHES "icpc$" AND NOT CMAKE_CXX_FLAGS MATCHES "-Wpedantic$")
    string(APPEND CMAKE_CXX_FLAGS " -Wpedantic")
  endif()
endif()

if (MSVC)
  # Disable warnings about using deprecated std::equal_to<>::result_type
  add_definitions (-D_SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING)
endif()

CHECK_FUNCTION_EXISTS(pow POW_FUNCTION_EXISTS)
if(NOT POW_FUNCTION_EXISTS)
  unset(POW_FUNCTION_EXISTS CACHE)
  list(APPEND CMAKE_REQUIRED_LIBRARIES m)
  CHECK_FUNCTION_EXISTS(pow POW_FUNCTION_EXISTS)
  if(POW_FUNCTION_EXISTS)
    set(MATH_LIBRARY m CACHE STRING "" FORCE)
  else()
    message(FATAL_ERROR "Failed making the pow() function available")
  endif()
endif()

# Code Coverage Configuration
add_library(cpp_coverage INTERFACE)

option(CODE_COVERAGE "Enable coverage reporting" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cpp_coverage INTERFACE -O0 -g --coverage)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
    target_link_options(cpp_coverage INTERFACE --coverage)
  else()
    target_link_libraries(cpp_coverage INTERFACE --coverage)
  endif()
endif()

enable_testing()

# Python
find_package(PythonInterp REQUIRED)
execute_process(
    COMMAND
    ${PYTHON_EXECUTABLE} -c [=[import os
import sysconfig
import sys
sys.stdout.write(os.path.dirname(sysconfig.get_config_h_filename()))
]=] OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)
find_package(PythonLibs REQUIRED)

# Boost
find_package(Boost 1.63 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# GSL
find_package(GSL 2.0 REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
if (WIN32)
  add_compile_definitions(GSL_DLL)
endif()

# Eigen3
find_package(Eigen3 3.3.1 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Googletest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF)
set(INSTALL_GMOCK OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest)
include_directories(
  ${gtest_SOURCE_DIR}/include
  ${gmock_SOURCE_DIR}/include
  ${gtest_SOURCE_DIR})

set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11)
add_subdirectory(src/pyinterp/core)
