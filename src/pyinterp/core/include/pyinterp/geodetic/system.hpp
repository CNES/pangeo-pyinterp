// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/pybind11.h>

#include "pyinterp/detail/geodetic/system.hpp"

namespace pyinterp::geodetic {

/// Wrapper
class System : public detail::geodetic::System {
 public:
  using detail::geodetic::System::System;

  /// Construction of the class from the base class.
  explicit System(detail::geodetic::System &&base)
      : detail::geodetic::System(base) {}

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(semi_major_axis(), flattening());
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple &state) -> System {
    if (state.size() != 2) {
      throw std::runtime_error("invalid state");
    }
    return {state[0].cast<double>(), state[1].cast<double>()};
  }
};

}  // namespace pyinterp::geodetic
