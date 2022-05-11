# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
cmake_minimum_required(VERSION 3.0)

include(CheckFunctionExists)
include(CheckCXXSourceRuns)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  message(FATAL_ERROR "The build directory must be different from the \
        root directory of this software.")
endif()

cmake_policy(SET CMP0048 NEW)
project(pyinterp LANGUAGES CXX)

if(POLICY CMP0063)
  cmake_policy(SET CMP0063 NEW)
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# CMake module search path
set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/tools;"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

# By default, build type is set to release, with debugging information.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

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

# Check if the C++ compiler and linker flags are set correctly.
macro(CHECK_CXX_COMPILER_AND_linker_flags result cxx_flags linker_flags)
  set(CMAKE_REQUIRED_FLAGS ${cxx_flags})
  set(CMAKE_REQUIRED_LIBRARIES ${linker_flags})
  set(CMAKE_REQUIRED_QUIET FALSE)
  check_cxx_source_runs("int main(int argc, char **argv) { return 0; }"
                        ${result})
  set(CMAKE_REQUIRED_FLAGS "")
  set(CMAKE_REQUIRED_LIBRARIES "")
  unset(result)
endmacro()

if(NOT WIN32)
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  find_package(Threads REQUIRED)
endif()

# Always use libc++ on Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  check_cxx_compiler_and_linker_flags(HAS_LIBCPP "-stdlib=libc++"
                                      "-stdlib=libc++")
  if(HAS_LIBCPP)
    string(APPEND CMAKEcxx_flags " -stdlib=libc++")
    string(APPEND CMAKE_EXE_linker_flags " -stdlib=libc++")
    string(APPEND CMAKE_SHARED_linker_flags " -stdlib=libc++")
    check_cxx_compiler_and_linker_flags(HAS_LIBCPPABI "-stdlib=libc++"
                                        "-stdlib=libc++ -lc++abi")
    if(HAS_LIBCPPABI)
      string(APPEND CMAKE_EXE_linker_flags " -lc++abi")
      string(APPEND CMAKE_SHARED_linker_flags " -lc++abi")
    endif()
  endif()
  check_cxx_compiler_and_linker_flags(HAS_SIZED_DEALLOCATION
                                      "-fsized-deallocation" "")
  if(HAS_SIZED_DEALLOCATION)
    string(APPEND CMAKEcxx_flags " -fsized-deallocation")
  endif()
endif()

if(NOT WIN32)
  if(NOT CMAKEcxx_flags MATCHES "-Wall$")
    string(APPEND CMAKEcxx_flags " -Wall")
  endif()
  if(NOT CMAKE_CXX_COMPILER MATCHES "icpc$" AND NOT CMAKEcxx_flags MATCHES
                                                "-Wpedantic$")
    string(APPEND CMAKEcxx_flags " -Wpedantic")
  endif()
endif()

if(MSVC)
  # Disable warnings about using deprecated std::equal_to<>::result_type
  add_definitions(-D_SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING)
  # Disable auto-linking and use cmake's dependency handling
  add_definitions(-DBOOST_ALL_NO_LIB)
endif()

check_function_exists(pow POW_FUNCTION_EXISTS)
if(NOT POW_FUNCTION_EXISTS)
  unset(POW_FUNCTION_EXISTS CACHE)
  list(APPEND CMAKE_REQUIRED_LIBRARIES m)
  check_function_exists(pow POW_FUNCTION_EXISTS)
  if(POW_FUNCTION_EXISTS)
    set(MATH_LIBRARY
        m
        CACHE STRING "" FORCE)
  else()
    message(FATAL_ERROR "Failed making the pow() function available")
  endif()
endif()

# Check if floating-point types fulfill the requirements of IEC 559 (IEEE 754)
# standard
macro(CHECK_FLOATING_POINT_IS_IEC559)
  message(STATUS "Performing Test HAVE_IEC559")
  file(
    WRITE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/is_iec559.cpp"
    "#include <limits>
int main() {
  return std::numeric_limits<double>::is_iec559 ? 1 : 0;
}")
  try_run(IS_IEC559 _UNUSED "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}"
          "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/is_iec559.cpp")
  if(IS_IEC559)
    message(STATUS "Performing Test HAVE_IEC559 - Success")
    add_definitions(-DHAVE_IEC559)
  else()
    message(STATUS "Performing Test HAVE_IEC559 - Failed")
  endif()
  unset(_UNUSED)
endmacro()

check_floating_point_is_iec559()

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

# Python
find_package(Python3 COMPONENTS Interpreter Development)

# Boost
find_package(Boost 1.63 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# We prefer the use of static libraries for the GSL.
if(CONDA_FORGE)
  set(__CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".so" ".dylib" ".lib")
endif()

# GSL
find_package(GSL 2.0 REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
if(WIN32)
  add_compile_definitions(GSL_DLL)
endif()

# We restore the suffixes of the libraries to be found.
if(CONDA_FORGE)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${__CMAKE_FIND_LIBRARY_SUFFIXES})
  unset(__CMAKE_FIND_LIBRARY_SUFFIXES)
endif()

# Blas
if(DEFINED ENV{MKLROOT})
  # First try to use MKL as a single dynamic library (conda-forge)
  set(BLA_VENDOR Intel10_64_dyn)
  find_package(BLAS)
  if(NOT BLAS_FOUND)
    # Otherwise try to use MKL lp64 model with sequential code
    set(BLA_VENDOR Intel10_64lp_seq)
    find_package(BLAS)
  endif()
endif()

if(BLAS_FOUND)
  # MKL
  if(DEFINED ENV{MKLROOT})
    find_path(
      MKL_INCLUDE_DIR
      NAMES mkl.h
      HINTS $ENV{MKLROOT}/include)
    if(MKL_INCLUDE_DIR)
      add_definitions(-DEIGEN_USE_MKL_ALL)
      add_definitions(-DMKL_LP64)
      include_directories(${MKL_INCLUDE_DIR})
    endif()
    # If a BLAS library has been found, it is used instead of the BLAS library
    # provided with GSL.
    set_target_properties(
      GSL::gsl
      PROPERTIES IMPORTED_LOCATION "${GSL_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${GSL_INCLUDE_DIRS}"
                 IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                 INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}")
  endif()
endif()

# Eigen3
find_package(Eigen3 3.3.1 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# GoogleTest
find_package(GTest)

set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11)
add_subdirectory(src/pyinterp/core)
