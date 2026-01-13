// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/temporal_axis.hpp"

namespace pyinterp::pybind {

/// @brief Temporal axis wrapper for datetime64 and timedelta64 types
class TemporalAxis : public math::TemporalAxis {
 public:
  /// @brief Build a TemporalAxis from its base class.
  /// @param[in] base_class Base class to copy
  explicit TemporalAxis(const math::TemporalAxis &base_class)
      : math::TemporalAxis(base_class) {}

  /// @brief Build a TemporalAxis from its base class.
  /// @param[in,out] base_class Base class to copy
  explicit TemporalAxis(math::TemporalAxis &&base_class)
      : math::TemporalAxis(std::move(base_class)) {}

  /// @brief Create a coordinate axis from datetime64 or timedelta64 values
  ///
  /// @param[in] points Axis values (datetime64 or timedelta64 array)
  /// @param[in] epsilon Comparison tolerance (in the same time units as dtype).
  /// Two timestamps/durations whose absolute difference is <= epsilon are
  /// treated as equal.
  /// @param[in] period The period of the axis for wrapping (timedelta64)
  explicit TemporalAxis(const nanobind::object &points,
                        const nanobind::object &epsilon = nanobind::none(),
                        const nanobind::object &period = nanobind::none());

  /// @brief Get the numpy dtype of this axis
  [[nodiscard]] auto dtype() const -> nanobind::object;

  /// @brief Get the first value of this axis
  [[nodiscard]] auto front() const -> nanobind::object;

  /// @brief Get the last value of this axis
  [[nodiscard]] auto back() const -> nanobind::object;

  /// @brief Get the minimum value of this axis.
  [[nodiscard]] auto min_value() const -> nanobind::object;

  /// @brief Get the maximum value of this axis.
  [[nodiscard]] auto max_value() const -> nanobind::object;

  /// @brief Get the increment (step) between values in this axis.
  /// @throw std::logic_error If this axis is not regular
  [[nodiscard]] auto increment() const -> nanobind::object;

  /// @brief Get the period of this axis.
  [[nodiscard]] auto period() const -> nanobind::object;

  /// @brief Get coordinate value at given index
  /// @param[in] index Index of the coordinate to retrieve
  /// @return Coordinate value at given index
  [[nodiscard]] auto coordinate_value(int64_t index) const -> nanobind::object;

  /// @brief Get coordinate values for a given slice
  /// @param[in] slice Slice of indexes to read
  /// @return Coordinate values
  [[nodiscard]] auto coordinate_values(const nanobind::slice &slice) const
      -> nanobind::object;

  /// @brief Find the axis element that contains the given coordinate position.
  /// @param[in] coordinates Coordinate positions to find
  /// @param[in] bounded Whether to bound the index to the axis range
  /// @return Index of the axis element that contains the given coordinate
  /// position
  [[nodiscard]] auto find_index(const nanobind::object &coordinates,
                                bool bounded) const -> Vector<int64_t>;

  /// @brief Find grid elements around the given coordinate position.
  /// @param[in] coordinates Coordinate positions to find
  /// @return Matrix of shape (n, 2) where the first column contains indexes i0
  /// and the second column contains indexes i1, or -1 if not found
  [[nodiscard]] auto find_indexes(const nanobind::object &coordinates) const
      -> Eigen::Matrix<int64_t, Eigen::Dynamic, 2, Eigen::RowMajor>;

  /// @brief Convert the input coordinates to the resolution managed by this
  /// instance.
  ///
  /// This function takes input coordinates, which may be in a different
  /// resolution, and converts them to the resolution that this instance is
  /// designed to handle. For instance, if this instance operates with
  /// datetime64 values in microsecond resolution and the input coordinates are
  /// provided in datetime64 hours, the output will be the input coordinates
  /// converted to datetime64 in microseconds.
  ///
  /// @param[in] coordinates Input datetime64 or timedelta64 array
  /// @return Converted coordinates in the resolution managed by this instance.
  [[nodiscard]] auto cast_to_temporal_axis(
      const nanobind::object &coordinates) const -> nanobind::object;

  /// @brief Convert the input coordinates to int64 representation.
  ///
  /// Like `cast_to_temporal_axis`, this function converts input coordinates to
  /// the resolution managed by this instance. However, instead of returning
  /// datetime64 or timedelta64 values, it returns their int64 representation,
  /// which is often used for internal computations.
  /// @param[in] coordinates Input datetime64 or timedelta64 array
  /// @return Converted coordinates as int64 array
  [[nodiscard]] auto cast_to_int64(const nanobind::object &coordinates) const
      -> Vector<int64_t>;

  /// @brief Get a tuple that fully encodes the state of this instance.
  ///
  /// @return Tuple containing the serialized state
  [[nodiscard]] auto getstate() const -> nanobind::tuple;

  /// @brief Create a new instance from a registered state.
  ///
  /// @param[in] state Tuple containing the serialized state
  /// @return New Axis instance
  /// @throw std::invalid_argument If the state is invalid or empty
  [[nodiscard]] static auto setstate(const nanobind::tuple &state)
      -> TemporalAxis;
};

}  // namespace pyinterp::pybind
