// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <sstream>
#include <stdexcept>
#include <string>

namespace pyinterp::detail {

/// Get a string representing the shape of a Eigen matrix.
///
/// @param array tensor to process
template <typename Array>
auto eigen_shape(const Array &array) -> std::string {
  std::stringstream ss;
  ss << "(" << array.rows() << ", " << array.cols() << ")";
  return ss.str();
}

/// Get a string representing the shape of a tensor.
///
/// @param array tensor to process
template <typename Array>
auto ndarray_shape(const Array &array) -> std::string {
  std::stringstream ss;
  ss << "(";
  for (auto ix = 0; ix < array.ndim(); ++ix) {
    ss << array.shape(ix) << ", ";
  }
  ss << ")";
  return ss.str();
}

/// Automation of vector size control to ensure that all vectors have the same
/// size.
///
/// @param name1 name of the variable containing the first vector
/// @param m1 first matrix
/// @param name2 name of the variable containing the second vector
/// @param m2 second matrix
/// @throw std::invalid_argument if the size of the two vectors is different
template <typename Matrix1, typename Matrix2>
void check_eigen_shape(const std::string &name1, const Matrix1 &m1,
                       const std::string &name2, const Matrix2 &m2) {
  if (m1.cols() != m2.cols() || m1.rows() != m2.rows()) {
    throw std::invalid_argument(name1 + ", " + name2 +
                                " could not be broadcast together with shape " +
                                eigen_shape(m1) + ", " + eigen_shape(m2));
  }
}

/// Vector size check function pattern.
///
/// @param name1 name of the variable containing the first vector
/// @param v1 first vector
/// @param name2 name of the variable containing the second vector
/// @param v2 second vector
/// @param args other vectors to be verified
/// @throw std::invalid_argument if the size of the two vectors is different
template <typename Matrix1, typename Matrix2, typename... Args>
void check_eigen_shape(const std::string &name1, const Matrix1 &v1,
                       const std::string &name2, const Matrix2 &v2,
                       Args... args) {
  static_assert(sizeof...(Args) % 2 == 0,
                "an even number of parameters is expected");
  check_eigen_shape(name1, v1, name2, v2);
  check_eigen_shape(name1, v1, args...);
}

/// Automation of vector size control to ensure that all vectors have the same
/// size.
///
/// @param name1 name of the variable containing the first vector
/// @param v1 first vector
/// @param name2 name of the variable containing the second vector
/// @param v2 second vector
/// @throw std::invalid_argument if the size of the two vectors is different
template <typename Vector1, typename Vector2>
void check_container_size(const std::string &name1, const Vector1 &v1,
                          const std::string &name2, const Vector2 &v2) {
  if (v1.size() != v2.size()) {
    throw std::invalid_argument(
        name1 + ", " + name2 + " could not be broadcast together with shape (" +
        std::to_string(v1.size()) + ", ) (" + std::to_string(v2.size()) +
        ", )");
  }
}

/// Vector size check function pattern.
///
/// @param name1 name of the variable containing the first vector
/// @param v1 first vector
/// @param name2 name of the variable containing the second vector
/// @param v2 second vector
/// @param args other vectors to be verified
/// @throw std::invalid_argument if the size of the two vectors is different
template <typename Vector1, typename Vector2, typename... Args>
void check_container_size(const std::string &name1, const Vector1 &v1,
                          const std::string &name2, const Vector2 &v2,
                          Args... args) {
  static_assert(sizeof...(Args) % 2 == 0,
                "an even number of parameters is expected");
  check_container_size(name1, v1, name2, v2);
  check_container_size(name1, v1, args...);
}

/// Automation of the control of the number of dimensions of a tensor.
///
/// @param name name of the variable containing the first array
/// @param ndim number of dimensions expected
/// @param a array to check
/// @throw std::invalid_argument if the number of dimensions of the table is
/// different from the expected one.
template <typename Array>
void check_array_ndim(const std::string &name, const int64_t ndim,
                      const Array &a) {
  if (a.ndim() != ndim) {
    throw std::invalid_argument(name + " must be a " + std::to_string(ndim) +
                                "-dimensional array");
  }
}

/// Control of the number of dimensions of a tensor.
///
/// @param name name of the variable containing the first array
/// @param ndim number of dimensions expected
/// @param a array to check
/// @throw std::invalid_argument if the number of dimensions of the table is
/// different from the expected one.
template <typename Array, typename... Args>
void check_array_ndim(const std::string &name, const int64_t ndim,
                      const Array &a, Args... args) {
  static_assert(sizeof...(Args) % 3 == 0,
                "number of parameters is expected to be a multiple of 3");
  check_array_ndim(name, ndim, a);
  check_array_ndim(args...);
}

/// Automation of array shape control to ensure that all tensors have the same
/// shape.
///
/// @param name1 name of the variable containing the first array
/// @param a1 first array
/// @param name2 name of the variable containing the second array
/// @param a2 second array
/// @throw std::invalid_argument if the shape of the two arrays is different
template <typename Array1, typename Array2>
void check_ndarray_shape(const std::string &name1, const Array1 &a1,
                         const std::string &name2, const Array2 &a2) {
  auto match = a1.ndim() == a2.ndim();
  if (match) {
    for (auto ix = 0; ix < a1.ndim(); ++ix) {
      if (a1.shape(ix) != a2.shape(ix)) {
        match = false;
        break;
      }
    }
  }
  if (!match) {
    throw std::invalid_argument(name1 + ", " + name2 +
                                " could not be broadcast together with shape " +
                                ndarray_shape(a1) + "  " + ndarray_shape(a2));
  }
}

/// Array shape check function pattern.
///
/// @param name1 name of the variable containing the first array
/// @param a1 first array
/// @param name2 name of the variable containing the second array
/// @param a2 second array
/// @throw std::invalid_argument if the shape of the two arrays is different
template <typename Array1, typename Array2, typename... Args>
void check_ndarray_shape(const std::string &name1, const Array1 &a1,
                         const std::string &name2, const Array2 &a2,
                         Args... args) {
  static_assert(sizeof...(Args) % 2 == 0,
                "an even number of parameters is expected");
  check_ndarray_shape(name1, a1, name2, a2);
  check_ndarray_shape(name1, a1, args...);
}

}  // namespace pyinterp::detail
