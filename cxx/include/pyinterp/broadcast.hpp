// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pyinterp::broadcast {

/// @brief Get a string representing the shape of an Eigen matrix.
/// @param[in] array Matrix to process
/// @return String in format "(rows, cols)"
template <typename Array>
[[nodiscard]] auto eigen_shape(const Array& array) -> std::string {
  return std::format("({}, {})", array.rows(), array.cols());
}

/// @brief Check that two Eigen matrices have the same shape.
/// @param[in] name1 Name of the first variable
/// @param[in] m1 First matrix
/// @param[in] name2 Name of the second variable
/// @param[in] m2 Second matrix
/// @throws std::invalid_argument if shapes differ
template <typename Matrix1, typename Matrix2>
constexpr auto check_eigen_shape(std::string_view name1, const Matrix1& m1,
                                 std::string_view name2, const Matrix2& m2)
    -> void {
  if (m1.cols() != m2.cols() || m1.rows() != m2.rows()) [[unlikely]] {
    throw std::invalid_argument(
        std::format("{}, {} could not be broadcast together with shape {}, {}",
                    name1, name2, eigen_shape(m1), eigen_shape(m2)));
  }
}

/// @brief Check that multiple Eigen matrices have the same shape (variadic
/// version).
/// @param[in] name1 Name of the first variable
/// @param[in] v1 First matrix
/// @param[in] name2 Name of the second variable
/// @param[in] v2 Second matrix
/// @param[in] args Additional (name, matrix) pairs to check
/// @throws std::invalid_argument if any shapes differ
template <typename Matrix1, typename Matrix2, typename... Args>
constexpr auto check_eigen_shape(std::string_view name1, const Matrix1& v1,
                                 std::string_view name2, const Matrix2& v2,
                                 Args&&... args) -> void {
  static_assert(sizeof...(Args) % 2 == 0,
                "an even number of parameters is expected");
  check_eigen_shape(name1, v1, name2, v2);
  check_eigen_shape(name1, v1, std::forward<Args>(args)...);
}

}  // namespace pyinterp::broadcast
