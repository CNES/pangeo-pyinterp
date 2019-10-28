// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <memory>
#include <pybind11/numpy.h>
#include "pyinterp/detail/axis.hpp"
#include "pyinterp/detail/math.hpp"

namespace pyinterp {

/// Implementation of the Python wrapper
class Axis;
class Axis : public detail::Axis, public std::enable_shared_from_this<Axis> {
 public:
  using detail::Axis::Axis;

  /// Create a coordinate axis from values.
  ///
  /// @param points axis values
  /// @param epsilon Maximum allowed difference between two real numbers in
  /// order to consider them equal.
  /// @param is_circle True, if the axis can represent a circle. Be careful,
  /// the angle shown must be expressed in degrees.
  /// @param is_radian True, if the coordinate system is radian.
  explicit Axis(pybind11::array_t<double, pybind11::array::c_style>& points,
                double epsilon, bool is_circle, bool is_radian);

  /// Get coordinate values.
  ///
  /// @param slice Slice of indexes to read
  /// @return coordinate values
  auto coordinate_values(const pybind11::slice& slice) const
      -> pybind11::array_t<double>;

  /// Given a coordinate position, find what axis element contains it.
  ///
  /// @param coordinate positions in this coordinate system
  /// @param bounded True if you want to obtain the closest value to an index
  ///   outside the axis definition range.
  /// @return A vector containing the indexes corresponding to the nearest
  ///   points on the axis or the value -1 if the *bounded* parameter is set
  ///   to false and the index looked for is located outside the limits of the
  ///   axis.
  auto find_index(const pybind11::array_t<double>& coordinates,
                  bool bounded) const -> pybind11::array_t<int64_t>;

  /// Get a tuple that fully encodes the state of this instance
  auto getstate() const -> pybind11::tuple;

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple& state) -> Axis;
};

}  // namespace pyinterp
