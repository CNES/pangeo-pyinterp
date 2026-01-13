// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <sys/stat.h>

#include <cstdint>
#include <optional>

#include "pyinterp/dateutils.hpp"
#include "pyinterp/math/axis.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::math {

/// @brief Temporal axis used to represent datetime64 or timedelta64 values.
class TemporalAxis : public Axis<int64_t> {
 public:
  /// @brief Default constructor
  TemporalAxis()
      : Axis<int64_t>(),
        dtype_(dateutils::DType(dateutils::DType::DateType::kDatetime64,
                                dateutils::DType::Resolution::kSecond)) {}

  /// @brief Construct a TemporalAxis from evenly spaced numbers.
  /// @param[in] dtype Data type used to encode datetime64 or timedelta64 values
  /// @param[in] start Start value of the axis
  /// @param[in] stop Stop value of the axis
  /// @param[in] num Number of points in the axis
  /// @param[in] epsilon Comparison tolerance (in the same time units as dtype).
  /// Two timestamps/durations whose absolute difference is <= epsilon are
  /// treated as equal.
  /// @param[in] period Period of the axis (optional)
  TemporalAxis(const dateutils::DType& dtype, const int64_t start,
               const int64_t stop, const size_t num, const int64_t epsilon,
               const std::optional<int64_t>& period = std::nullopt)
      : Axis<int64_t>(start, stop, num, epsilon, period), dtype_(dtype) {}

  /// @brief Construct a TemporalAxis from a set of points.
  /// @param[in] dtype Data type used to encode datetime64 or timedelta64 values
  /// @param[in] points Vector of points
  /// @param[in] epsilon Comparison tolerance (in the same time units as dtype).
  /// Two timestamps/durations whose absolute difference is <= epsilon are
  /// treated as equal.
  /// @param[in] period Period of the axis (optional)
  TemporalAxis(const dateutils::DType& dtype,
               const Eigen::Ref<const Vector<int64_t>>& points,
               const int64_t epsilon,
               const std::optional<int64_t>& period = std::nullopt)
      : Axis<int64_t>(points, epsilon, period), dtype_(dtype) {}

  /// @brief Construct a TemporalAxis from an Axis<int64_t>
  /// @param[in] axis Axis instance
  /// @param[in] dtype Data type used to encode datetime64 or timedelta64 values
  TemporalAxis(Axis<int64_t> axis, const dateutils::DType& dtype)
      : Axis<int64_t>(std::move(axis)), dtype_(dtype) {}

  /// @brief Get the data type used to encode datetime64 or timedelta64 values
  [[nodiscard]] constexpr auto dtype() const -> const dateutils::DType& {
    return dtype_;
  }

  /// @brief Check if two TemporalAxis objects are equal
  [[nodiscard]] auto operator==(const TemporalAxis& other) const -> bool {
    return dtype_ == other.dtype_ &&
           static_cast<math::Axis<int64_t>>(*this) ==
               static_cast<math::Axis<int64_t>>(other);
  }

  /// @brief Check if two TemporalAxis objects are not equal
  [[nodiscard]] auto operator!=(const TemporalAxis& other) const -> bool {
    return !(*this == other);
  }

  /// @brief Get string representation of a coordinate value
  /// @param[in] value Coordinate value
  /// @return String representation of the coordinate value
  [[nodiscard]] auto coordinate_repr(const int64_t value) const
      -> std::string final {
    return TemporalAxis::format_temporal(value, dtype_);
  }

  /// @brief Get string representation of this axis.
  ///
  /// @return String representation of this axis
  [[nodiscard]] explicit operator std::string() const override {
    // Call the base-class conversion operator directly to avoid virtual
    // recursion.
    auto result = Axis<int64_t>::operator std::string();
    // Replace "Axis" by "TemporalAxis[<dtype>]"
    result.replace(
        0, 4,
        std::format("TemporalAxis[{}]", static_cast<std::string>(dtype_)));
    return result;
  }

  /// @brief Get the serialized state of this instance.
  ///
  /// @return Tuple containing the serialized state
  /// @throw std::runtime_error If the axis handler type is unknown
  [[nodiscard]] auto pack() const -> serialization::Writer final {
    auto state = Axis<int64_t>::pack();
    state.write(std::string(dtype_));
    return state;
  }

  /// @brief Create a new instance from a registered state.
  ///
  /// @param[in] state Tuple containing the serialized state
  /// @return New Axis instance
  /// @throw std::invalid_argument If the state is invalid or empty
  [[nodiscard]] static auto unpack(serialization::Reader& state)
      -> TemporalAxis {
    Axis<int64_t> axis = Axis<int64_t>::unpack(state);
    std::string dtype_str = state.read_string();
    dateutils::DType dtype(dtype_str.c_str());
    return {std::move(axis), dtype};
  }

 private:
  /// Data type used to encode datetime64 or timedelta64 values.
  dateutils::DType dtype_;

  /// @brief Format a temporal scalar value according to the dtype
  /// @param[in] value Scalar value to format
  /// @param[in] dtype Data type used to encode datetime64 or timedelta64 values
  /// @return Formatted string
  static inline auto format_datetime(int64_t value,
                                     const dateutils::DType& dtype)
      -> std::string {
    return dateutils::datetime64_to_string(value, dtype);
  }

  /// @brief Format a temporal timedelta value according to the dtype
  /// @param[in] value Timedelta value to format
  /// @param[in] dtype Data type used to encode timedelta64 values
  /// @return Formatted string
  static inline auto format_timedelta(int64_t value,
                                      const dateutils::DType& dtype)
      -> std::string {
    return dateutils::timedelta64_to_string(value, dtype);
  }

  /// @brief Format a temporal value according to the dtype
  /// @param[in] value Value to format
  /// @param[in] dtype Data type used to encode datetime64 or timedelta64 values
  /// @return Formatted string
  static inline auto format_temporal(int64_t value,
                                     const dateutils::DType& dtype)
      -> std::string {
    return dtype.datetype() == dateutils::DType::DateType::kDatetime64
               ? TemporalAxis::format_datetime(value, dtype)
               : TemporalAxis::format_timedelta(value, dtype);
  }

  /// @brief Get string representation of a scalar value
  /// @param[in] value Scalar value
  /// @return String representation of the scalar value
  [[nodiscard]] auto scalar_repr(const int64_t value) const
      -> std::string final {
    return dtype_.datetype() == dateutils::DType::DateType::kDatetime64
               ? format_datetime(value, dtype_)
               : std::format("{}", value);
  }

  /// @brief Get string representation of the period of this axis.
  ///
  /// @return String representation of the period
  [[nodiscard]] auto period_repr() const -> std::string final {
    return format_timedelta(this->period().value_or(0), dtype_);
  }

  /// @brief Get string representation of the increment of this axis.
  ///
  /// @return String representation of the increment
  [[nodiscard]] auto increment_repr() const -> std::string final {
    return format_timedelta(this->increment(), dtype_);
  }
};

}  // namespace pyinterp::math
