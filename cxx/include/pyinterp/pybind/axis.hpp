// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <ranges>

#include "pyinterp/math/axis.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"

namespace pyinterp::pybind {

/// Python wrapper for Axis with numeric types
///
/// @tparam T Numeric type
template <typename T>
  requires std::is_arithmetic_v<T>
class Axis : public math::Axis<T> {
 public:
  using math::Axis<T>::Axis;

  /// @brief Build an Axis from the base class
  /// @param[in] base_class Base class to copy
  explicit Axis(const math::Axis<T> &base_class) : math::Axis<T>(base_class) {}

  /// @brief Build an Axis from the base class
  /// @param[in,out] base_class Base class to copy
  explicit Axis(math::Axis<T> &&base_class)
      : math::Axis<T>(std::move(base_class)) {}

  /// @brief Get coordinate values.
  ///
  /// @param[in] slice Slice of indexes to read
  /// @return Coordinate values
  /// @throw pybind11::error_already_set If slice computation fails
  [[nodiscard]] auto coordinate_values(const nanobind::slice &slice) const
      -> Vector<T> {
    auto [start, stop, step, slicelength] = slice.compute(this->size());
    auto result = Vector<T>(slicelength);

    // Use ranges for cleaner iteration
    auto indices = std::views::iota(size_t{0}, slicelength) |
                   std::views::transform(
                       [start, step](size_t ix) { return start + ix * step; });

    size_t ix = 0;
    for (auto idx : indices) {
      result(ix++) = (*this)(idx);
    }
    return result;
  }

  /// @brief Find the axis element that contains the given coordinate position.
  ///
  /// @param[in] coordinates Positions in this coordinate system
  /// @param[in] bounded If false, returns the closest value to an index outside
  /// the axis definition range
  /// @return Vector containing the indexes corresponding to the nearest points
  /// on the axis, or -1 if bounded is false and the coordinate is outside the
  /// axis limits
  [[nodiscard]] auto find_index(const Eigen::Ref<const Vector<T>> &coordinates,
                                bool bounded) const -> Vector<int64_t> {
    auto size = coordinates.size();
    auto result = Vector<int64_t>(size);

    for (auto ix : std::views::iota(int64_t{0}, static_cast<int64_t>(size))) {
      result(ix) = math::Axis<T>::find_index(coordinates(ix), bounded);
    }
    return result;
  }

  /// @brief Find grid elements around the given coordinate position.
  ///
  /// This method finds i0 and i1 such that (*this)(i0) <= coordinate <
  /// (*this)(i1).
  /// @param[in] coordinates Positions in this coordinate system
  /// @return Matrix of shape (n, 2) where the first column contains indexes i0
  /// and the second column contains indexes i1, or -1 if not found
  [[nodiscard]] auto find_indexes(
      const Eigen::Ref<const Vector<T>> &coordinates) const
      -> Eigen::Matrix<int64_t, Eigen::Dynamic, 2, Eigen::RowMajor> {
    auto size = coordinates.size();
    auto result =
        Eigen::Matrix<int64_t, Eigen::Dynamic, 2, Eigen::RowMajor>(size, 2);

    for (auto ix : std::views::iota(int64_t{0}, size)) {
      if (auto indexes = math::Axis<T>::find_indexes(coordinates(ix))) {
        const auto [i0, i1] = *indexes;
        result(ix, 0) = i0;
        result(ix, 1) = i1;
      } else {
        result.row(ix).setConstant(-1);
      }
    }
    return result;
  }

  /// @brief Get a tuple that fully encodes the state of this instance.
  ///
  /// @return Tuple containing the serialized state
  [[nodiscard]] auto getstate() const -> nanobind::tuple {
    serialization::Writer state;
    {
      nanobind::gil_scoped_release release;
      state = math::Axis<T>::pack();
    }
    return nanobind::make_tuple(writer_to_ndarray(std::move(state)));
  }

  /// @brief Create a new instance from a registered state.
  ///
  /// @param[in] state Tuple containing the serialized state
  /// @return New Axis instance
  /// @throw std::invalid_argument If the state is invalid or empty
  [[nodiscard]] static auto setstate(const nanobind::tuple &state) -> Axis<T> {
    if (state.size() != 1) {
      throw std::invalid_argument("Invalid state");
    }
    auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
    auto reader = reader_from_ndarray(array);
    {
      nanobind::gil_scoped_release release;
      return Axis<T>(std::move(math::Axis<T>::unpack(reader)));
    }
  }
};

/// @brief Initialize the core Axis class
/// @param[in] m Module in which to insert the class
void init_axis(nanobind::module_ &m);

}  // namespace pyinterp::pybind
