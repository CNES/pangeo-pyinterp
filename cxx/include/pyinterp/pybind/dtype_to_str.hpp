// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <optional>

namespace pyinterp::pybind {

/// @brief Convert a dtype parameter from Python to a string
/// @param[in] dtype The dtype parameter from Python. This can be a string,
/// numpy.dtype, or numpy type.
/// @return Optional string representing the dtype, or std::nullopt if dtype is
/// None
inline auto dtype_to_str(const nanobind::object& dtype)
    -> std::optional<std::string> {
  std::optional<std::string> dtype_str;
  if (!dtype.is_none()) {
    try {
      // Try direct string cast first
      dtype_str = nanobind::cast<std::string>(dtype);
    } catch (...) {
      // If not a string, try to get the dtype.name attribute (for np.dtype
      // objects)
      try {
        auto name_attr = dtype.attr("name");
        dtype_str = nanobind::cast<std::string>(name_attr);
      } catch (...) {
        // If that fails, try __name__ attribute (for type objects like
        // np.float64)
        try {
          auto name_attr = dtype.attr("__name__");
          dtype_str = nanobind::cast<std::string>(name_attr);
        } catch (...) {
          throw std::invalid_argument(
              "dtype must be a string, numpy.dtype, or numpy type");
        }
      }
    }
  }
  return dtype_str;
}

}  // namespace pyinterp::pybind
