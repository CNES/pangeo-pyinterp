// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math.hpp"
#include "pyinterp/math/axis/container.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::math {
namespace axis {
/// Type of boundary handling on an Axis.
enum Boundary : uint8_t {
  kExpand,  //!< Expand the boundary as a constant.
  kShrink,  //!< Shrink the boundary to fit the data.
  kSym,     //!< Symmetrical boundary conditions.
  kUndef,   //!< Boundary violation is not defined.
  kWrap,    //!< Circular boundary conditions.
};

/// Result type for windowed index searches.
///
/// Contains a vector of indexes surrounding a coordinate and the pair of
/// indexes that bracket the coordinate at the window center.
/// - First element: Vector of indexes (size = 2*half_window_size), or empty
///   if boundary constraints cannot be satisfied
/// - Second element: Pair of (lower_index, upper_index) that bracket the
///   coordinate, where indices are relative to the window (range [0,
///   window_size-1])
using IndexWindow =
    std::pair<std::vector<int64_t>, std::pair<int64_t, int64_t>>;

}  // namespace  axis

/// A coordinate axis specifies one of the coordinates of a variable's values.
///
/// An Axis can represent either regular (evenly-spaced) or irregular coordinate
/// values. It supports optional periodicity for cyclic dimensions (e.g.,
/// longitude, time of day).
///
/// The axis automatically chooses the appropriate storage strategy:
/// - Regular: For evenly-spaced values (optimized memory usage)
/// - Irregular: For arbitrary spaced values
/// - Undefined: For empty or uninitialized axes
///
/// @tparam T Arithmetic type of data handled by this Axis (e.g., float, double,
/// int)
/// @note All axis values must be monotonic (strictly increasing or decreasing)
template <typename T>
  requires std::is_arithmetic_v<T>
class Axis {
 public:
  /// Type of values stored in this axis.
  using value_type = T;

  /// @brief Default constructor
  Axis() = default;

  /// @brief Create a coordinate axis from evenly spaced numbers.
  ///
  /// Creates an axis with evenly spaced values over a specified interval.
  /// @param[in] start The first value of the axis
  /// @param[in] stop The last value of the axis
  /// @param[in] num Number of samples in the axis
  /// @param[in] epsilon Maximum allowed difference between two numbers to
  /// consider them equal
  /// @param[in] period Period of the axis (e.g., 360 for degrees, 24 for
  /// hours). If not set, the axis is non-periodic
  Axis(const T start, const T stop, const size_t num, const T epsilon,
       const std::optional<T>& period = std::nullopt);

  /// @brief Create a coordinate axis from a vector of values.
  ///
  /// @param[in] values Axis values
  /// @param[in] epsilon Maximum allowed difference between two real numbers to
  /// consider them equal
  /// @param[in] period Period of the axis (e.g., 360 for degrees, 24 for
  /// hours). If not set, the axis is non-periodic
  Axis(const Eigen::Ref<const Vector<T>>& values, T epsilon,
       const std::optional<T>& period = std::nullopt);

  /// @brief Destructor
  virtual ~Axis() = default;

  /// @brief Copy constructor
  ///
  /// @param[in] rhs Right-hand side value
  Axis(const Axis& rhs)
      : is_periodic_(rhs.is_periodic_),
        period_(rhs.period_),
        container_(rhs.clone()) {}

  /// @brief Move constructor
  ///
  /// @param[in,out] rhs Right-hand side value
  Axis(Axis&& rhs) noexcept
      : is_periodic_(rhs.is_periodic_),
        period_(rhs.period_),
        container_(std::move(rhs.container_)) {}

  /// @brief Copy assignment operator
  ///
  /// @param[in] rhs Right-hand side value
  /// @return Reference to this object
  auto operator=(const Axis& rhs) -> Axis& {
    if (this != &rhs) {
      is_periodic_ = rhs.is_periodic_;
      period_ = rhs.period_;
      container_ = rhs.clone();
    }
    return *this;
  }

  /// @brief Move assignment operator
  ///
  /// @param[in,out] rhs Right-hand side value
  /// @return Reference to this object
  auto operator=(Axis&& rhs) noexcept -> Axis& {
    if (this != &rhs) {
      is_periodic_ = rhs.is_periodic_;
      period_ = rhs.period_;
      container_ = std::move(rhs.container_);
    }
    return *this;
  }

  /// @brief Create an Axis from a container (for internal use).
  ///
  /// This constructor allows direct initialization from a container object,
  /// typically used during deserialization or internal construction.
  /// @param[in] container The axis container (Regular, Irregular, or Undefined)
  /// @param[in] is_periodic True if the axis is periodic
  /// @param[in] period The period value; must be positive for periodic axes
  Axis(std::unique_ptr<axis::Abstract<T>> container, const bool is_periodic,
       const std::optional<T>& period)
      : is_periodic_(is_periodic),
        period_(period.has_value() ? period.value() : T{}),
        container_(std::move(container)) {}

  /// @brief Get the ith coordinate value.
  ///
  /// @param[in] index Coordinate index (between 0 and size()-1 inclusive)
  /// @return Coordinate value
  /// @throw std::out_of_range If index is not in range [0, size() - 1]
  [[nodiscard]] constexpr auto coordinate_value(const int64_t index) const
      -> T {
    if (index < 0 || index >= size()) {
      throw std::out_of_range("axis index out of range");
    }
    return container_->coordinate_value(index);
  }

  /// @brief Get a slice of the axis.
  ///
  /// @param[in] start Index of the first element to include in the slice
  /// @param[in] count Number of elements to include in the slice
  /// @return A slice of the axis
  /// @throw std::out_of_range If the slice exceeds axis bounds
  [[nodiscard]] constexpr auto slice(int64_t start, int64_t count) const
      -> Vector<T> {
    if (start < 0 || start + count > size()) {
      throw std::out_of_range("axis index out of range");
    }
    return container_->slice(start, count);
  };

  /// @brief Get the minimum coordinate value.
  ///
  /// @return Minimum coordinate value
  [[nodiscard]] constexpr auto min_value() const noexcept -> T {
    return container_->min_value();
  }

  /// @brief Get the maximum coordinate value.
  ///
  /// @return Maximum coordinate value
  [[nodiscard]] constexpr auto max_value() const noexcept -> T {
    return container_->max_value();
  }

  /// @brief Get the number of values for this axis.
  ///
  /// @return The number of values
  [[nodiscard]] constexpr auto size() const noexcept -> int64_t {
    return container_->size();
  }

  /// @brief Check if this axis values are spaced regularly.
  ///
  /// @return True if the axis is regular
  [[nodiscard]] constexpr auto is_regular() const noexcept -> bool {
    return container_->type() == axis::AxisType::kRegular;
  }

  /// @brief Check if this axis is periodic.
  ///
  /// @return True if the axis is periodic
  [[nodiscard]] constexpr auto is_periodic() const noexcept -> bool {
    return is_periodic_;
  }

  /// @brief Get the period of this axis.
  ///
  /// @return The period value
  [[nodiscard]] constexpr auto period() const noexcept -> std::optional<T> {
    return is_periodic_ ? std::make_optional(period_) : std::nullopt;
  }

  /// @brief Get the first value of this axis.
  ///
  /// @return The first value
  [[nodiscard]] constexpr auto front() const noexcept -> T {
    return container_->front();
  }

  /// @brief Get the last value of this axis.
  ///
  /// @return The last value
  [[nodiscard]] constexpr auto back() const noexcept -> T {
    return container_->back();
  }

  /// @brief Test if the data is sorted in ascending order.
  ///
  /// @return True if the data is sorted in ascending order
  [[nodiscard]] constexpr auto is_ascending() const noexcept -> bool {
    return container_->is_ascending();
  }

  /// @brief Reverse the order of elements in this axis.
  auto flip() -> void { container_->flip(); }

  /// @brief Get increment value if is_regular().
  ///
  /// @return Increment value
  /// @throw std::logic_error If this instance does not represent a regular axis
  [[nodiscard]] constexpr auto increment() const -> T {
    if (container_->type() != axis::AxisType::kRegular) {
      throw std::logic_error(
          "Increment cannot be retrieved because the axis is not regular.");
    }
    return static_cast<axis::Regular<T>*>(container_.get())->step();
  }

  /// @brief Compare two axis instances for equality.
  ///
  /// @param[in] rhs Another axis to compare
  /// @return True if axes are equal
  constexpr auto operator==(Axis const& rhs) const -> bool {
    // Compare period first; container equality does not account for periodicity
    return period_ == rhs.period_ && *container_ == *rhs.container_;
  }

  /// @brief Compare two axis instances for inequality.
  ///
  /// @param[in] rhs Another axis to compare
  /// @return True if axes are not equal
  constexpr auto operator!=(Axis const& rhs) const -> bool {
    return !this->operator==(rhs);
  }

  /// @brief Get the ith coordinate value.
  ///
  /// @param[in] index Coordinate index (between 0 and size()-1 inclusive)
  /// @return Coordinate value
  constexpr auto operator()(const int64_t index) const -> T {
    return container_->coordinate_value(index);
  }

  /// @brief Normalize a coordinate value with respect to the axis definition.
  ///
  /// @param[in] coordinate Position in this coordinate system
  /// @param[in] min Minimum value of the axis
  /// @return Normalized coordinate value
  [[nodiscard]] constexpr auto normalize_coordinate(const T coordinate,
                                                    const T min) const noexcept
      -> T {
    if (period_ != T(0) && (coordinate >= min + period_ || coordinate < min)) {
      return math::normalize_period<T>(coordinate, min, period_);
    }
    return coordinate;
  }

  /// @brief Normalize a coordinate value with respect to the axis definition.
  ///
  /// @param[in] coordinate Position in this coordinate system
  /// @return Normalized coordinate value
  [[nodiscard]] constexpr auto normalize_coordinate(
      const T coordinate) const noexcept -> T {
    return normalize_coordinate(coordinate, min_value());
  }

  /// @brief Search for the index corresponding to the requested value.
  ///
  /// @param[in] coordinate Position in this coordinate system
  /// @param[in] bounded If false, returns -1 if the value is outside the
  /// coordinate system; otherwise returns the first or last index
  /// @return Index of the requested value, or -1 if outside the coordinate
  /// system area
  [[nodiscard]] constexpr auto find_index(T coordinate,
                                          const bool bounded) const -> int64_t {
    if (Fill<T>::is_fill_value(coordinate)) {
      return -1;
    }
    coordinate = normalize_coordinate(coordinate);
    auto result = container_->find_index(coordinate, bounded);
    if (result == -1 && is_periodic_) {
      // Handle periodic case
      return (coordinate - max_value()) < (min_value() + period_ - coordinate)
                 ? this->size() - 1
                 : 0;
    }
    return result;
  }

  /// @brief Search for the indexes that surround the requested value.
  ///
  /// Finds the two adjacent axis indexes that bracket the given coordinate.
  /// For periodic axes, wraps around the boundaries when appropriate.
  /// @param[in] coordinate Position in this coordinate system
  /// @return Tuple (lower_index, upper_index) if the coordinate is bracketed,
  /// std::nullopt if outside the axis range (or wrapped for periodic axes)
  [[nodiscard]] auto find_indexes(T coordinate) const
      -> std::optional<std::pair<int64_t, int64_t>>;

  /// @brief Retrieve a range of indexes surrounding the specified coordinate.
  ///
  /// Returns a symmetric window of indexes centered on the bracketing indexes.
  /// For example, with coordinate at indexes [10, 11] and half_window_size=3,
  /// returns [8, 9, 10, 11, 12, 13] (total size: 2*half_window_size).
  ///
  /// @param[in] coordinate Position in this coordinate system
  /// @param[in] half_window_size Number of indexes to retrieve on each side
  /// of the coordinate; result size is (2*half_window_size)
  /// @param[in] boundary Type of boundary handling (Expand, Wrap, Sym, or
  /// Undef)
  /// @return An optional pair if the boundary constraints can be satisfied:
  ///   - Vector of indexes (size = 2*half_window_size), or empty if boundary
  ///     constraints cannot be satisfied
  ///   - Pair of (lower_index, upper_index) bracketing the coordinate at the
  ///     window center. For 'Shrink' boundary mode, these may differ from
  ///     (half_window_size - 1, half_window_size); otherwise they equal those
  ///     values
  [[nodiscard]] auto find_indexes(T coordinate, size_t half_window_size,
                                  axis::Boundary boundary) const
      -> std::optional<axis::IndexWindow>;

  /// @brief Get a string representation of a coordinate handled by this axis.
  ///
  /// @param[in] value Value to be converted to string
  /// @return String representation of the value
  [[nodiscard]] virtual inline auto coordinate_repr(const T value) const
      -> std::string {
    return std::format("{}", value);
  }

  /// @brief Get a string representation of this axis.
  ///
  /// @return String representation of this axis
  [[nodiscard]] explicit virtual operator std::string() const;

  /// @brief Serialize the axis state for storage or transmission.
  ///
  /// Encodes all axis properties including type, values/range, periodicity,
  /// and period value into a serialization buffer.
  /// @return Serialized state as a Writer object
  /// @throw std::runtime_error If the axis container type is unknown
  [[nodiscard]] virtual auto pack() const -> serialization::Writer;

  /// @brief Deserialize an axis from serialized state.
  ///
  /// Reconstructs an Axis instance from a serialization buffer created by
  /// getstate(). Automatically restores the correct container type and all
  /// properties.
  /// @param[in] state Reference to serialization Reader containing encoded axis
  /// data
  /// @return New Axis instance with restored properties
  /// @throw std::invalid_argument If the state is invalid, empty, or contains
  /// an unrecognized container type identifier
  [[nodiscard]] static auto unpack(serialization::Reader& state) -> Axis<T>;

  /// @brief Check if that coordinate is within the axis bounds.
  ///
  /// @param[in] coordinate Position in this coordinate system
  /// @return True if the coordinate is within the axis bounds
  [[nodiscard]] constexpr auto contains(const T coordinate) const noexcept
      -> bool {
    if (Fill<T>::is_fill_value(coordinate)) {
      return false;
    }

    // If periodic, all coordinates are considered within bounds
    if (is_periodic_) {
      return true;
    }
    return coordinate >= min_value() && coordinate <= max_value();
  }

 protected:
  /// @brief Get the axis container.
  ///
  /// @return The axis container
  [[nodiscard]] constexpr auto container() const noexcept
      -> const std::unique_ptr<axis::Abstract<T>>& {
    return container_;
  }

  /// @brief Get a string representation of a scalar value handled by this axis.
  ///
  /// @param[in] value Value to be converted to string
  /// @return String representation of the value
  [[nodiscard]] virtual inline auto scalar_repr(const T value) const
      -> std::string {
    return coordinate_repr(value);
  }

  /// @brief Get a string representation of the period of this axis.
  ///
  /// @return String representation of the period
  [[nodiscard]] virtual inline auto period_repr() const -> std::string {
    return std::format("{}", period_);
  }

  /// @brief Get a string representation of the increment of this axis.
  ///
  /// @return String representation of the increment
  [[nodiscard]] virtual inline auto increment_repr() const -> std::string {
    return std::format("{}", increment());
  }

 private:
  /// Function pointer to handle boundary violations
  using BoundaryHandler = int64_t (*)(int64_t, int64_t);

  /// Magic number for axis serialization
  static constexpr uint32_t kMagicNumber = 0x41584953;

  /// True, if the axis is periodic.
  bool is_periodic_{false};

  /// The value of the period of the axis.
  T period_{};

  /// The object that handles access and searches for the values defined by
  /// the axis.
  std::unique_ptr<axis::Abstract<T>> container_{
      std::make_unique<axis::Undefined<T>>()};

  /// @brief Clone the axis container.
  ///
  /// @return A cloned copy of the axis container
  [[nodiscard]] auto clone() const -> std::unique_ptr<axis::Abstract<T>>;

  /// @brief Determine if provided points form a regular (evenly-spaced)
  /// sequence.
  ///
  /// Checks if points can be represented as a linear sequence by comparing
  /// them against a mathematically generated linspace.
  /// @param[in] points Axis points to analyze
  /// @param[in] epsilon Maximum allowed difference to consider values equal
  /// @return The uniform step size if points are evenly spaced, std::nullopt
  /// otherwise
  [[nodiscard]] static auto is_evenly_spaced(
      const Eigen::Ref<const Vector<T>>& points, const T epsilon)
      -> std::optional<T>;

  /// @brief Validate and compute axis properties after container creation.
  ///
  /// Validates axis invariants (non-empty, monotonic), and adjusts the periodic
  /// flag based on whether the range matches the specified period.
  /// @param[in] epsilon Maximum allowed difference to consider values equal
  /// @throw std::invalid_argument If axis is empty, non-monotonic, or violates
  /// period constraints
  void compute_properties(const T epsilon);

  /// @brief Adjust periodic axis values to maintain continuity across period
  /// boundaries.
  ///
  /// For periodic axes with values that cross the period boundary, this method
  /// adds/subtracts the period to maintain monotonicity. For example, longitude
  /// values [170, 180, -170, -160] are adjusted to [170, 180, 190, 200].
  /// @param[in] points Axis points to adjust
  /// @return Adjusted points maintaining monotonicity, or std::nullopt if
  /// already monotonic or fewer than 2 points
  [[nodiscard]]
  auto adjust_period(const Vector<T>& points) -> std::optional<Vector<T>>;

  /// @brief Create and initialize the appropriate container type.
  ///
  /// Determines whether values are regular or irregular and instantiates
  /// the corresponding container type (Regular or Irregular).
  /// @param[in] values Axis values (will be moved)
  /// @param[in] epsilon Maximum allowed difference to consider values equal;
  /// used to determine if spacing is regular
  void create_container_impl(Vector<T>&& values, T epsilon);

  /// @brief Create the appropriate container to represent the axis.
  ///
  /// @param[in] values Axis values
  /// @param[in] epsilon Maximum allowed difference between two real numbers to
  /// consider them equal
  inline void create_container(const Eigen::Ref<const Vector<T>>& values,
                               T epsilon) {
    create_container_impl(values, epsilon);
  }

  /// @brief Create the appropriate container to represent the axis based on the
  /// provided values.
  /// @param[in] values Axis values
  /// @param[in] epsilon Maximum allowed difference between two real numbers in
  /// order to consider them equal.
  inline void create_container(Vector<T>&& values, T epsilon) {
    create_container_impl(std::move(values), epsilon);
  }

  /// @brief Create a boundary handler function for index wrapping.
  ///
  /// Returns a function that handles out-of-bounds index wrapping according to
  /// the specified boundary mode: Expand (clamp), Wrap (modulo), Sym
  /// (symmetric), or Undef (return -1). For periodic axes, always uses modulo
  /// wrapping.
  ///
  /// @param[in] boundary Boundary handling mode (ignored if axis is periodic)
  /// @param[in] is_periodic True to use periodic (modulo) wrapping
  /// @return Function pointer taking (index, size) and returning wrapped index
  /// or -1 if boundary violation cannot be handled
  static constexpr auto make_boundary_handler(axis::Boundary boundary,
                                              bool is_periodic) noexcept
      -> BoundaryHandler;

  /// @brief Format a list of values for string representation.
  /// @tparam Formatter Type of the value formatting function
  /// @param size Number of values
  /// @param format_value Function to format a value at given index
  /// @return Formatted string
  [[nodiscard]] auto format_value_list() const -> std::string;
};

// ============================================================================
// Implementation
// ============================================================================
template <typename T>
  requires std::is_arithmetic_v<T>
auto Axis<T>::clone() const -> std::unique_ptr<axis::Abstract<T>> {
  switch (container_->type()) {
    case axis::AxisType::kRegular:
      return std::make_unique<axis::Regular<T>>(
          *static_cast<const axis::Regular<T>*>(container_.get()));
    case axis::AxisType::kIrregular:
      return std::make_unique<axis::Irregular<T>>(
          *static_cast<const axis::Irregular<T>*>(container_.get()));
    case axis::AxisType::kUndefined:
      return std::make_unique<axis::Undefined<T>>(
          *static_cast<const axis::Undefined<T>*>(container_.get()));
  }
  // This should never be reached, but added for completeness
  std::unreachable();
}

// ============================================================================
template <typename T>
  requires std::is_arithmetic_v<T>
auto Axis<T>::is_evenly_spaced(const Eigen::Ref<const Vector<T>>& points,
                               const T epsilon) -> std::optional<T> {
  auto n = points.size();
  if (n < 2) {
    // The axis contains a single value
    return std::nullopt;
  }

  const auto step = (points[n - 1] - points[0]) / static_cast<T>(n - 1);
  const auto diff =
      (points - Vector<T>::LinSpaced(n, points[0], points[n - 1])).cwiseAbs();
  if (diff.maxCoeff() <= epsilon) {
    return step;
  }
  return std::nullopt;
}

// ============================================================================
template <typename T>
  requires std::is_arithmetic_v<T>
void Axis<T>::compute_properties(const T epsilon) {
  if (container_->size() == 0) {
    throw std::invalid_argument("Cannot create an axis with no values.");
  }
  if (!container_->is_monotonic()) {
    throw std::invalid_argument("Axis values must be monotonic.");
  }
  // If the axis is periodic, determine if it represents a full period
  if (is_periodic_) {
    if (container_->type() == axis::AxisType::kRegular) {
      auto step = static_cast<axis::Regular<T>*>(container_.get())->step();
      is_periodic_ = math::is_same(static_cast<T>(std::fabs(step * size())),
                                   period_, epsilon);
    } else {
      auto step = (container_->back() - container_->front()) /
                  static_cast<T>(container_->size() - 1);
      is_periodic_ =
          std::abs((container_->max_value() - container_->min_value()) -
                   period_) <= step + epsilon;
    }
  }
}

// ============================================================================

template <typename T>
  requires std::is_arithmetic_v<T>
auto Axis<T>::adjust_period(const Vector<T>& points)
    -> std::optional<Vector<T>> {
  // Determine if the points are monotonic
  const auto n = points.size();
  if (n < 2) {
    return std::nullopt;
  }
  const auto ascending = points[0] < points[1];

  // Check if the axis is monotonic
  auto needs_adjustment = false;
  for (int64_t ix = 1; ix < n && !needs_adjustment; ++ix) {
    needs_adjustment =
        ascending ? points[ix - 1] > points[ix] : points[ix - 1] < points[ix];
  }

  if (!needs_adjustment) {
    return std::nullopt;
  }

  // Create adjusted copy
  auto result = Vector<T>(points);
  bool cross = false;

  for (int64_t ix = 1; ix < n; ++ix) {
    if (!cross) {
      cross =
          ascending ? result[ix - 1] > result[ix] : result[ix - 1] < result[ix];
    }
    if (cross) {
      result[ix] += ascending ? period_ : -period_;
    }
  }
  return result;
}

// ============================================================================

template <typename T>
  requires std::is_arithmetic_v<T>
void Axis<T>::create_container_impl(Vector<T>&& values, T epsilon) {
  auto step = is_evenly_spaced(values, epsilon);
  if (step) {
    container_.reset(new axis::Regular<T>(values[0], values[values.size() - 1],
                                          static_cast<size_t>(values.size())));
  } else {
    container_.reset(new axis::Irregular<T>(std::move(values)));
  }
}

// ============================================================================

template <typename T>
  requires std::is_arithmetic_v<T>
Axis<T>::Axis(const T start, const T stop, const size_t num, const T epsilon,
              const std::optional<T>& period)
    : is_periodic_(period.has_value()),
      period_(period ? period.value() : T{}),
      container_(std::make_unique<axis::Regular<T>>(start, stop, num)) {
  // Adjust the periodicity property based on the axis values and epsilon
  compute_properties(epsilon);
}

// ///////////////////////////////////////////////////////////////////////////

template <typename T>
  requires std::is_arithmetic_v<T>
Axis<T>::Axis(const Eigen::Ref<const Vector<T>>& values, T epsilon,
              const std::optional<T>& period)
    : is_periodic_(period.has_value()), period_(period ? period.value() : T{}) {
  if (values.size() > std::numeric_limits<int64_t>::max()) {
    throw std::invalid_argument(
        "The axis size exceeds the maximum allowable limit of " +
        std::to_string(std::numeric_limits<int64_t>::max()) + " elements.");
  }

  if (period_) {
    auto adjusted = adjust_period(values);
    if (adjusted) {
      create_container(std::move(*adjusted), epsilon);
    } else {
      create_container(values, epsilon);
    }
  } else {
    create_container(values, epsilon);
  }
  compute_properties(epsilon);
}

// ============================================================================

template <typename T>
  requires std::is_arithmetic_v<T>
auto Axis<T>::find_indexes(T coordinate) const
    -> std::optional<std::pair<int64_t, int64_t>> {
  if (Fill<T>::is_fill_value(coordinate)) {
    return std::nullopt;
  }
  const auto container_size = size();
  coordinate = normalize_coordinate(coordinate);

  // Cache container pointer to avoid repeated virtual calls
  const auto* container_ptr = container_.get();
  // Search for the nearest index
  auto i0 = container_ptr->find_index(coordinate, false);

  if (i0 == -1) {
    // The requested coordinate is outside the axis range. If the axis is
    // periodic, return the indexes that wrap around the axis.
    return is_periodic_
               ? std::make_optional(std::make_pair(container_size - 1, 0))
               : std::nullopt;
  }

  // Calculate the difference between the requested coordinate and the axis
  // value at index i0
  const auto delta = coordinate - container_ptr->coordinate_value(i0);

  // Early exit for exact match
  if (delta == 0) [[unlikely]] {
    // Special case when the axis contains a single value
    if (container_size == 1) [[unlikely]] {
      // The axis contains a single value, cannot frame the coordinate
      return std::nullopt;
    }
    const auto i1 = (i0 == container_size - 1) ? i0 - 1 : i0 + 1;
    return std::make_optional(
        std::make_pair(std::min(i0, i1), std::max(i0, i1)));
  }

  auto i1 = i0;
  const bool ascending = is_ascending();

  if (delta < 0) {
    i0 = ascending ? i0 - 1 : i0 + 1;
  } else {
    i1 = ascending ? i0 + 1 : i0 - 1;
  }

  // Handle periodic wrapping
  if (is_periodic_) {
    return std::make_optional(
        std::make_pair(math::remainder(i0, container_size),
                       math::remainder(i1, container_size)));
  }

  return (i0 >= 0 && i1 >= 0 && i0 < container_size && i1 < container_size)
             ? std::make_optional(std::make_pair(i0, i1))
             : std::nullopt;
}

// ============================================================================

template <typename T>
  requires std::is_arithmetic_v<T>
constexpr auto Axis<T>::make_boundary_handler(axis::Boundary boundary,
                                              bool is_periodic) noexcept
    -> BoundaryHandler {
  if (is_periodic) {
    return [](int64_t idx, int64_t size) { return math::remainder(idx, size); };
  }

  switch (boundary) {
    case axis::kExpand:
      return [](int64_t idx, int64_t size) {
        return std::clamp(idx, static_cast<int64_t>(0), size - 1);
      };
    case axis::kWrap:
      return
          [](int64_t idx, int64_t size) { return math::remainder(idx, size); };
    case axis::kSym:
      return [](int64_t idx, int64_t size) {
        // Symmetric boundary logic
        if (idx < 0) {
          return math::remainder(-idx, size);
        }
        return size - 2 - math::remainder(idx - size, size);
      };
    default:
      return [](int64_t, int64_t) { return static_cast<int64_t>(-1); };
  }
}

// ============================================================================

template <typename T>
  requires std::is_arithmetic_v<T>
auto Axis<T>::find_indexes(T coordinate, size_t half_window_size,
                           axis::Boundary boundary) const
    -> std::optional<axis::IndexWindow> {
  if (half_window_size == 0) {
    return {};
  }
  auto indexes = find_indexes(coordinate);
  if (!indexes) {
    // If the requested coordinate cannot be framed, check if the axis is a
    // singleton. For a singleton axis, the result is valid only if the
    // requested coordinate matches the single value stored in the axis.
    if (this->size() != 1) {
      return {};
    }
    return std::make_pair(std::vector<int64_t>(half_window_size << 1U, 0),
                          std::make_pair(0, 0));
  }
  // Length of the axis
  const auto container_size = this->size();

  // Function to handle boundary conditions
  const auto handle_boundary =
      Axis<T>::make_boundary_handler(boundary, is_periodic_);

  // Use a deque to efficiently push to the front and back
  auto result = std::deque<int64_t>();
  result.push_back(std::get<0>(*indexes));
  result.push_back(std::get<1>(*indexes));

  // Center indexes in the result window
  auto center_indexes = std::pair<int64_t, int64_t>(0, 1);

  // Offset in relation to the first indexes found
  size_t shift = 1;

  // Construction of window indexes based on the initial indexes found
  while (shift < half_window_size) {
    int64_t before = std::get<0>(*indexes) - shift;
    if (before < 0) {
      before = handle_boundary(before, container_size);
    }
    if (before >= 0) {
      result.push_front(before);
      // Shift center indexes to the right
      center_indexes.first++;
      center_indexes.second++;
    } else if (boundary != axis::kShrink) {
      // Boundary violation cannot be handled. Return an empty result.
      return {};
    }

    int64_t after = std::get<1>(*indexes) + shift;
    if (after >= container_size) {
      after = handle_boundary(after, container_size);
    }
    if (after >= 0) {
      result.push_back(after);
    } else if (boundary != axis::kShrink) {
      // Boundary violation cannot be handled. Return an empty result.
      return {};
    }
    ++shift;
  }
  return std::make_pair(std::vector<int64_t>(result.begin(), result.end()),
                        center_indexes);
}

// ============================================================================

template <typename T>
  requires std::is_arithmetic_v<T>
[[nodiscard]] auto Axis<T>::format_value_list() const -> std::string {
  std::string result = "  values: [";
  constexpr size_t max_display = 6;
  const auto size = static_cast<size_t>(this->size());

  if (size <= max_display) {
    for (size_t i = 0; i < size; ++i) {
      if (i > 0) {
        result += ", ";
      }
      result += coordinate_repr((*this)(i));
    }
  } else {
    // Show first 3 and last 3 with ellipsis
    for (size_t i = 0; i < 3; ++i) {
      if (i > 0) {
        result += ", ";
      }
      result += coordinate_repr((*this)(i));
    }
    result += ", ...";
    for (size_t i = size - 3; i < size; ++i) {
      result += ", " + coordinate_repr((*this)(i));
    }
  }

  return result + std::format("]\n  size: {}", size);
}

// ============================================================================

template <typename T>
  requires std::is_arithmetic_v<T>
Axis<T>::operator std::string() const {
  std::string result = std::format(
      "Axis({}{})\n", is_regular() ? "regular" : "irregular",
      is_periodic() ? std::format(", period={}", period_repr()) : "");

  if (this->is_regular()) {
    result += std::format(
        "  range: [{}, {}]\n"
        "  step: {}\n"
        "  size: {}",
        scalar_repr(min_value()), scalar_repr(max_value()), increment_repr(),
        size());
  } else {
    result += format_value_list();
  }

  return result;
}

// ============================================================================

template <typename T>
  requires std::is_arithmetic_v<T>
auto Axis<T>::pack() const -> serialization::Writer {
  serialization::Writer buffer;
  // Write magic number for validation
  buffer.write(kMagicNumber);

  // Serialize the axis container type identifier (axis::AxisType) as a single
  // byte.
  buffer.write(static_cast<uint8_t>(container_->type()));

  // Regular axis
  if (container_->type() == axis::AxisType::kRegular) {
    const auto* regular_axis =
        static_cast<const axis::Regular<T>*>(container_.get());
    buffer.write(regular_axis->front());
    buffer.write(regular_axis->back());
    buffer.write(static_cast<size_t>(regular_axis->size()));
    buffer.write(is_periodic_);
    buffer.write(period_);
  }
  // Irregular axis
  else if (container_->type() == axis::AxisType::kIrregular) {
    const auto* irregular_axis =
        static_cast<const axis::Irregular<T>*>(container_.get());
    buffer.write(irregular_axis->points());
    buffer.write(is_periodic_);
    buffer.write(period_);
  }
  // Undefined axis has no additional state to serialize
  return buffer;
}

// ============================================================================

template <typename T>
  requires std::is_arithmetic_v<T>
auto Axis<T>::unpack(serialization::Reader& state) -> Axis<T> {
  if (state.size() == 0) {
    throw std::invalid_argument("Cannot restore axis from empty state.");
  }
  auto magic_number = state.read<uint32_t>();
  if (magic_number != kMagicNumber) {
    throw std::invalid_argument("Invalid axis state (bad magic number).");
  }

  auto type_id = state.read<uint8_t>();
  if (type_id == static_cast<uint8_t>(axis::AxisType::kRegular)) {
    const auto start = state.read<T>();
    const auto stop = state.read<T>();
    const auto size = state.read<size_t>();
    const auto is_periodic = state.read<bool>();
    const auto period = state.read<T>();
    return Axis<T>(std::make_unique<axis::Regular<T>>(start, stop, size),
                   is_periodic,
                   is_periodic ? std::make_optional(period) : std::nullopt);
  } else if (type_id == static_cast<uint8_t>(axis::AxisType::kIrregular)) {
    const auto points = state.read_eigen<T>();
    const auto is_periodic = state.read<bool>();
    const auto period = state.read<T>();
    return Axis<T>(std::make_unique<axis::Irregular<T>>(std::move(points)),
                   is_periodic,
                   is_periodic ? std::make_optional(period) : std::nullopt);
  } else if (type_id == static_cast<uint8_t>(axis::AxisType::kUndefined)) {
    return Axis<T>();
  }
  throw std::invalid_argument("Cannot restore axis from invalid state.");
}

}  // namespace pyinterp::math
