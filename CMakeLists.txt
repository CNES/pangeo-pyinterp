# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
cmake_minimum_required(VERSION 3.10)

# ##############################################################################
# Import CMake modules
# ##############################################################################
include(CheckFunctionExists)
include(CheckCXXSourceRuns)

# ##############################################################################
# Check if the source directory is different from the build directory This is to
# prevent in-source builds.
# ##############################################################################
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  message(FATAL_ERROR "The build directory must be different from the \
        root directory of this software.")
endif()

# ##############################################################################
# Configure additional paths for CMake to search for custom modules
# ##############################################################################
set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/tools;"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

# ##############################################################################
# Project settings
# ##############################################################################
project(pyinterp LANGUAGES CXX)

# ##############################################################################
# CMake policies
# ##############################################################################
if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()

if(POLICY CMP0063)
  cmake_policy(SET CMP0063 NEW)
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY_CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

# ##############################################################################
# Configure compiler flags and options
# ##############################################################################

# Define the build type Asan to enable address sanitizer if the compiler matches
# the requirements (GCC or Clang)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(SANITIZE "address,undefined")
  if(UNIX AND NOT APPLE)
    set(SANITIZE "${SANITIZE},leak")
  endif()

  set(CMAKE_C_FLAGS_ASAN
      "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fsanitize=${SANITIZE} \
      -fno-omit-frame-pointer -fno-common"
      CACHE STRING "" FORCE)
  set(CMAKE_CXX_FLAGS_ASAN
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=${SANITIZE} \
      -fno-omit-frame-pointer -fno-common"
      CACHE STRING "" FORCE)
  set(CMAKE_EXE_LINKER_FLAGS_ASAN
      "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=${SANITIZE}"
      CACHE STRING "" FORCE)
  set(CMAKE_SHARED_LINKER_FLAGS_ASAN
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=${SANITIZE}"
      CACHE STRING "" FORCE)
endif()

set_property(
  CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                  "RelWithDebInfo" "ASan")

# By default, build type is set to release, with debugging information.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# The library must be built using C++20 compiler.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MACOSX_RPATH 1)

include(CheckCXXCompilerFlag)
if(NOT WIN32)
  check_cxx_compiler_flag("-std=c++20" HAS_CPP20_FLAG)
else()
  check_cxx_compiler_flag("/std:c++20" HAS_CPP20_FLAG)
endif()
if(NOT HAS_CPP20_FLAG)
  message(FATAL_ERROR "Unsupported compiler -- requires C++20 support!")
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

# Always use libc++ on Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  check_cxx_compiler_and_linker_flags(HAS_LIBCPP "-stdlib=libc++"
                                      "-stdlib=libc++")
  if(HAS_LIBCPP)
    string(APPEND CMAKE_CXX_FLAGS " -stdlib=libc++")
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
    string(APPEND CMAKE_CXX_FLAGS " -fsized-deallocation")
  endif()
endif()

if(NOT WIN32)
  if(NOT CMAKE_CXX_FLAGS MATCHES "-Wall$")
    string(APPEND CMAKE_CXX_FLAGS " -Wall")
  endif()
  if(NOT CMAKE_CXX_COMPILER MATCHES "icpc$" AND NOT CMAKE_CXX_FLAGS MATCHES
                                                "-Wpedantic$")
    string(APPEND CMAKE_CXX_FLAGS " -Wpedantic")
  endif()

  # Suppress warnings on ARM64 NEON code
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    # Suppress psabi warning due to Boost Geometry ABI changes
    string(APPEND CMAKE_CXX_FLAGS " -Wno-psabi")
    # Suppress class-memaccess warning in Eigen NEON packet math
    string(APPEND CMAKE_CXX_FLAGS " -Wno-class-memaccess")
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

# ##############################################################################
# Configure the build options
# ##############################################################################
option(CODE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)

# ##############################################################################
# Code Coverage Configuration
# ##############################################################################
add_library(cpp_coverage INTERFACE)

if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cpp_coverage INTERFACE -O0 -g --coverage)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
    target_link_options(cpp_coverage INTERFACE --coverage)
  else()
    target_link_libraries(cpp_coverage INTERFACE --coverage)
  endif()
endif()

# ##############################################################################
# Configure the project dependencies
# ##############################################################################

# Threads
if(NOT WIN32)
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  find_package(Threads REQUIRED)
endif()

# Python
add_definitions(-DPYBIND11_ASSERT_GIL_HELD_INCREF_DECREF)
find_package(Python3 COMPONENTS Interpreter Development)

# Boost
find_package(Boost 1.79 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# BLAS
#
# If the environment variable MKLROOT is defined, we will try to use MKL as a
# dynamic library first. If it fails, we will try to use MKL lp64 model with
# sequential code. If MKL is not found, we will try to use OpenBLAS or Apple
# Accelerate framework. If none of them is found, we will use the generic
# implementation of BLAS provided by Eigen.
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
  endif()
else()
  set(BLA_VENDOR_LIST "Apple" "OpenBLAS" "Generic")
  foreach(item IN LISTS BLA_VENDOR_LIST)
    set(BLA_VENDOR ${item})
    find_package(BLAS)
    if(BLAS_FOUND)
      break()
    endif()
  endforeach()
  if(BLAS_FOUND)
    add_definitions(-DEIGEN_USE_BLAS)
  else()
    message(
      WARNING "No BLAS library found. Eigen uses its own BLAS implementation.")
    set(BLAS_LIBRARIES "")
  endif()
endif()

# Eigen3
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)

# GoogleTest
include(CTest)
find_package(GTest)

# ##############################################################################
# Enable clang-tidy if requested
# ##############################################################################
if(ENABLE_CLANG_TIDY)
  find_program(
    CLANG_TIDY_EXE
    NAMES "clang-tidy"
    DOC "/usr/bin/clang-tidy")
  if(NOT CLANG_TIDY_EXE)
    message(
      FATAL_ERROR
        "clang-tidy not found. Please set CLANG_TIDY_EXE to clang-tidy "
        "executable.")
  endif()
  string(
    CONCAT
      CLANG_TIDY_CMD
      "clang-tidy;-checks=-*,boost-*,concurrency-*,modernize-*,performance-*,"
      "clang-analyzer-*,portability-*,-portability-simd-intrinsics,google-*,"
      ";-fix")
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_CMD}")
  unset(CLANG_TIDY_EXE CACHE)
  unset(CLANG_TIDY_CMD CACHE)
  message(STATUS "clang-tidy enabled.")
endif()

# When building Python bindings with a static library, we need to set symbol
# visibility to hidden to prevent symbol conflicts in the Python module. This
# avoids collisions between symbols from different libraries when the Python
# extension is loaded.
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ##############################################################################
# Add subdirectories
# ##############################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11)
add_subdirectory(src/pyinterp/core)

# ##############################################################################
# Print configuration summary
# ##############################################################################

# Helper function to format boolean option
function(format_enabled value_ result_var)
  if(value_)
    set(${result_var}
        "ON"
        PARENT_SCOPE)
  else()
    set(${result_var}
        "OFF"
        PARENT_SCOPE)
  endif()
endfunction()

format_enabled(CODE_COVERAGE CODE_COVERAGE_LINE)
format_enabled(ENABLE_CLANG_TIDY CLANG_TIDY_LINE)

set(CONFIG_LINES
    "==================[ PyInterp Configuration Summary ]=================="
    "Platform:               ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}"
    "Build type:             ${CMAKE_BUILD_TYPE}"
    "CMake Version:          ${CMAKE_VERSION}"
    "C++ flags:              ${CMAKE_CXX_FLAGS}"
    "C++ Standard Required:  ${CMAKE_CXX_STANDARD_REQUIRED}"
    "C++ Standard:           ${CMAKE_CXX_STANDARD}"
    "Linker flags:           ${CMAKE_EXE_LINKER_FLAGS}"
    "Compiler ID:            ${CMAKE_CXX_COMPILER_ID}"
    "Compiler Version:       ${CMAKE_CXX_COMPILER_VERSION}"
    "Boost include dir:      ${Boost_INCLUDE_DIRS}"
    "Eigen3 include dir:     ${EIGEN3_INCLUDE_DIR}"
    "BLAS:                   ${BLA_VENDOR}"
    "Code Coverage:          ${CODE_COVERAGE_LINE}"
    "Clang-Tidy:             ${CLANG_TIDY_LINE}"
    "Python3:                ${Python3_VERSION}"
    "======================================================================")

foreach(line ${CONFIG_LINES})
  message(STATUS "${line}")
endforeach()

unset(CODE_COVERAGE_LINE)
unset(CLANG_TIDY_LINE)
unset(CONFIG_LINES)
