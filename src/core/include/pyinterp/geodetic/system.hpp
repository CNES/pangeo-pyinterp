#pragma once
#include "pyinterp/detail/geodetic/system.hpp"
#include <pybind11/pybind11.h>

namespace pyinterp {
namespace geodetic {

/// Wrapper
class System : public detail::geodetic::System {
 public:
  using detail::geodetic::System::System;

  /// Construction of the class from the base class.
  explicit System(detail::geodetic::System&& base)
      : detail::geodetic::System(base){};

  /// Get a tuple that fully encodes the state of this instance
  pybind11::tuple getstate() const {
    return pybind11::make_tuple(semi_major_axis(), flattening());
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static System setstate(const pybind11::tuple& state) {
    if (state.size() != 2) {
      throw std::runtime_error("invalid state");
    }
    return System(state[0].cast<double>(), state[1].cast<double>());
  }
};

}  // namespace geodetic
}  // namespace pyinterp