// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pyinterp/detail/axis/container.hpp"
#include "pyinterp/detail/math.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp::axis {

/// Type of boundary handling on an Axis.
enum Boundary : uint8_t {
  kExpand,  //!< Expand the boundary as a constant.
  kWrap,    //!< Circular boundary conditions.
  kSym,     //!< Symmetrical boundary conditions.
  kUndef,   //!< Boundary violation is not defined.
};

}  // namespace pyinterp::axis

namespace pyinterp::detail {

/// A coordinate axis is a Variable that specifies one of the coordinates
/// of a variable's values.
///
/// @tparam T Type of data handled by this Axis
template <typename T>
class Axis {
 public:
  /// Default constructor
  Axis() = default;

  /// Create a coordinate axis from evenly spaced numbers over a specified
  /// interval.
  ///
  /// @param start the first value of the axis
  /// @param stop the last value of the axis
  /// @param num number of samples in the axis
  /// @param epsilon Maximum allowed difference between two numbers in
  /// order to consider them equal.
  /// @param is_circle True, if the axis can represent a circle.
  Axis(const T start, const T stop, const T num, const T epsilon,
       const bool is_circle)
      : circle_(is_circle ? T(360) : math::Fill<T>::value()),
        axis_(std::make_shared<axis::container::Regular<T>>(
            axis::container::Regular<T>(start, stop, num))) {
    compute_properties(epsilon);
  }

  /// Create a coordinate axis from values.
  ///
  /// @param values Axis values
  /// @param epsilon Maximum allowed difference between two real numbers in
  /// order to consider them equal.
  /// @param is_circle True, if the axis can represent a circle.
  explicit Axis(const Eigen::Ref<const Vector<T>> &values, T epsilon,
                bool is_circle)
      : circle_(is_circle ? T(360) : math::Fill<T>::value()) {
    // Axis size control
    if (values.size() > std::numeric_limits<int64_t>::max()) {
      throw std::invalid_argument(
          "The size of the axis must not contain more than " +
          std::to_string(std::numeric_limits<int64_t>::max()) + "elements.");
    }

    if (is_angle()) {
      auto normalized_values = normalize_longitude(values);
      normalized_values
          ? initialize_from_values(*normalized_values, epsilon, true)
          : initialize_from_values(values, epsilon, false);
    } else {
      initialize_from_values(values, epsilon, false);
    }
  }

  /// Destructor
  virtual ~Axis() = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Axis(const Axis &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Axis(Axis &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Axis &rhs) -> Axis & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Axis &&rhs) noexcept -> Axis & = default;

  /// Get the ith coordinate value.
  ///
  /// @param index which coordinate. Between 0 and size()-1 inclusive
  /// @return coordinate value
  /// @throw std::out_of_range if index in not in range [0, size() - 1].
  [[nodiscard]] constexpr auto coordinate_value(const int64_t index) const
      -> T {
    if (index < 0 || index >= size()) {
      throw std::out_of_range("axis index out of range");
    }
    return axis_->coordinate_value(index);
  }

  /// Get a slice of the axis.
  ///
  /// @param start index of the first element to include in the slice
  /// @param count number of elements to include in the slice
  /// @return a slice of the axis
  [[nodiscard]] constexpr auto slice(int64_t start, int64_t count) const
      -> Vector<T> {
    if (start < 0 || start + count > size()) {
      throw std::out_of_range("axis index out of range");
    }
    return axis_->slice(start, count);
  };

  /// Get the minimum coordinate value.
  ///
  /// @return minimum coordinate value
  [[nodiscard]] constexpr auto min_value() const -> T {
    return axis_->min_value();
  }

  /// Get the maximum coordinate value.
  ///
  /// @return maximum coordinate value
  [[nodiscard]] constexpr auto max_value() const -> T {
    return axis_->max_value();
  }

  /// Get the number of values for this axis
  ///
  /// @return the number of values
  [[nodiscard]] constexpr auto size() const noexcept -> int64_t {
    return axis_->size();
  }

  /// Check if this axis values are spaced regularly
  [[nodiscard]] constexpr auto is_regular() const noexcept -> bool {
    return dynamic_cast<axis::container::Regular<T> *>(axis_.get()) != nullptr;
  }

  /// Returns true if this axis represents a circle.
  [[nodiscard]] constexpr auto is_circle() const noexcept -> bool {
    return is_circle_;
  }

  /// Does the axis represent an angle?
  ///
  /// @return true if the axis represent an angle
  [[nodiscard]] constexpr auto is_angle() const noexcept -> bool {
    return math::Fill<T>::is_not(circle_);
  }

  /// Get the first value of this axis
  ///
  /// @return the first value
  [[nodiscard]] constexpr auto front() const -> T { return axis_->front(); }

  /// Get the last value of this axis
  ///
  /// @return the last value
  [[nodiscard]] constexpr auto back() const -> T { return axis_->back(); }

  /// Test if the data is sorted in ascending order.
  ///
  /// @return True if the data is sorted in ascending order.
  [[nodiscard]] constexpr auto is_ascending() const -> bool {
    return axis_->is_ascending();
  }

  /// Reverse the order of elements in this axis
  auto flip() -> void { axis_->flip(); }

  /// Get increment value if is_regular()
  ///
  /// @return increment value if is_regular()
  /// @throw std::logic_error if this instance does not represent a regular axis
  [[nodiscard]] constexpr auto increment() const -> T {
    auto ptr = dynamic_cast<axis::container::Regular<T> *>(axis_.get());
    if (ptr == nullptr) {
      throw std::logic_error("this axis is not regular.");
    }
    return ptr->step();
  }

  /// compare two variables instances
  ///
  /// @param rhs an other axis to compare
  /// @return if axis are equals
  constexpr auto operator==(Axis const &rhs) const -> bool {
    return *axis_ == *rhs.axis_ && is_circle_ == rhs.is_circle_;
  }

  /// compare two variables instances
  ///
  /// @param rhs an other axis to compare
  /// @return if axis are equals
  constexpr auto operator!=(Axis const &rhs) const -> bool {
    return !this->operator==(rhs);
  }

  /// Get the ith coordinate value.
  ///
  /// @param index which coordinate. Between 0 and size()-1 inclusive
  /// @return coordinate value
  constexpr auto operator()(const int64_t index) const -> T {
    return axis_->coordinate_value(index);
  }

  /// Returns the normalized value of the coordinate with the respect to the
  /// axis definition.
  [[nodiscard]] constexpr auto normalize_coordinate(const T coordinate,
                                                    const T min) const noexcept
      -> T {
    if (is_angle() && (coordinate >= min + circle_ || coordinate < min)) {
      return math::normalize_angle(coordinate, min, circle_);
    }
    return coordinate;
  }

  /// Returns the normalized value with respect to the axis definition. This
  /// means if the axis defines a circle, this method returns a value within the
  /// interval [font(), back()] otherwise it returns the value supplied.
  [[nodiscard]] constexpr auto normalize_coordinate(
      const T coordinate) const noexcept -> T {
    return normalize_coordinate(coordinate, axis_->min_value());
  }

  /// Given a coordinate position, find what axis element contains it.
  ///
  /// @param coordinate position in this coordinate system
  /// @param bounded if true, returns "-1" if the value is located outside this
  /// coordinate system, otherwise the value of the first element if the value
  /// is located before, or the value of the last element of this container if
  /// the requested value is located after.
  /// @return index of the grid point containing it or -1 if outside grid area
  [[nodiscard]] constexpr auto find_index(const T coordinate,
                                          const bool bounded) const -> int64_t {
    return axis_->find_index(normalize_coordinate(coordinate), bounded);
  }

  /// Given a coordinate position, find what axis element contains it.
  ///
  /// This function is identical to the "index_function" except that the values
  /// located between the last and the first point of the axis representing a
  /// circle will not return the value -1.
  ///
  /// @param coordinate position in this coordinate system
  /// @param bounded if true, returns "-1" if the value is located outside this
  /// coordinate system, otherwise the value of the first element if the value
  /// is located before, or the value of the last element of this container if
  /// the requested value is located after.
  /// @return index of the grid point containing it or -1 if outside grid area
  [[nodiscard]] constexpr auto find_nearest_index(T coordinate,
                                                  const bool bounded) const
      -> int64_t {
    coordinate = normalize_coordinate(coordinate);
    auto result = axis_->find_index(coordinate, bounded);
    if (result == -1 && is_circle_) {
      result = (coordinate - max_value()) < (min_value() + 360 - coordinate)
                   ? this->size() - 1
                   : 0;
    }
    return result;
  }

  /// Given a coordinate position, find grids elements around it.
  /// This mean that
  /// @code
  /// (*this)(i0) <= coordinate < (*this)(i1)
  /// @endcode
  ///
  /// @param coordinate position in this coordinate system
  /// @return None if coordinate is outside the axis definition domain otherwise
  /// the tuple (i0, i1)
  [[nodiscard]] auto find_indexes(T coordinate) const
      -> std::optional<std::tuple<int64_t, int64_t>> {
    coordinate = normalize_coordinate(coordinate);
    auto length = size();
    auto i0 = find_index(coordinate, false);

    /// If the value is outside the circle, then the value is between the last
    /// and first index.
    if (i0 == -1) {
      return is_circle_ ? std::make_tuple(static_cast<int64_t>(length - 1), 0LL)
                        : std::optional<std::tuple<int64_t, int64_t>>();
    }

    // Given the delta between the found coordinate and the given coordinate,
    // chose the other index that frames the coordinate
    auto delta = coordinate - (*this)(i0);
    auto i1 = i0;
    if (delta == 0) {
      // The requested coordinate is located on an element of the axis.
      i1 == length - 1 ? --i0 : ++i1;
    } else {
      if (delta < 0) {
        // The found point is located after the coordinate provided.
        is_ascending() ? --i0 : ++i0;
        if (is_circle_) {
          i0 = math::remainder(i0, length);
        }
      } else {
        // The found point is located before the coordinate provided.
        is_ascending() ? ++i1 : --i1;
        if (is_circle_) {
          i1 = math::remainder(i1, length);
        }
      }
    }

    if (i0 >= 0 && i0 < length && i1 >= 0 && i1 < length) {
      return std::make_tuple(i0, i1);
    }
    return std::optional<std::tuple<int64_t, int64_t>>{};
  }

  /// Create a table of "size" indices located on either side of the required
  /// position.
  ///
  /// @param coordinate Position in this coordinate system
  /// @param size Size of the half window to be built.
  /// @param boundary How to handle boundaries (this parameter is not used if
  /// the manipulated axis is a circle.)
  /// @return A table of size "2*size" containing the indices of the axis
  /// framing the value provided or an empty table if the value is located
  /// outside the axis definition domain.
  [[nodiscard]] auto find_indexes(T coordinate, uint32_t size,
                                  ::pyinterp::axis::Boundary boundary) const
      -> std::vector<int64_t> {
    if (size == 0) {
      throw std::invalid_argument("The size must not be zero.");
    }

    // Searches the initial indexes and populate the result
    auto indexes = find_indexes(coordinate);
    if (!indexes) {
      // If it's not possible to frame the requested coordinate, we check if the
      // given axis is a singleton. In this case, the result is defined as long
      // as the requested coordinate is equal to the value stored in the axis.
      if (size == 1 && coordinate == (*this)(0)) {
        return std::vector<int64_t>(size << 1U, 0);
      }
      return {};
    }

    // Axis size
    auto len = this->size();

    auto result = std::vector<int64_t>(size << 1U);
    std::tie(result[size - 1], result[size]) = *indexes;

    // Offset in relation to the first indexes found
    uint32_t shift = 1;

    // Construction of window indexes based on the initial indexes found
    while (shift < size) {
      int64_t before = std::get<0>(*indexes) - shift;
      if (before < 0) {
        if (!is_circle_) {
          switch (boundary) {
            case ::pyinterp::axis::kExpand:
              before = 0;
              break;
            case ::pyinterp::axis::kWrap:
              before = math::remainder(len + before, len);
              break;
            case ::pyinterp::axis::kSym:
              before = math::remainder(-before, len);
              break;
            default:
              return {};
          }
        } else {
          before = math::remainder(before, len);
        }
      }
      int64_t after = std::get<1>(*indexes) + shift;
      if (after >= len) {
        if (!is_circle_) {
          switch (boundary) {
            case ::pyinterp::axis::kExpand:
              after = len - 1;
              break;
            case ::pyinterp::axis::kWrap:
              after = math::remainder(after, len);
              break;
            case ::pyinterp::axis::kSym:
              after = len - 2 - math::remainder(after - len, len);
              break;
            default:
              return {};
          }
        } else {
          after = math::remainder(after, len);
        }
      }
      result[size - shift - 1] = before;
      result[size + shift] = after;
      ++shift;
    }
    return result;
  }

  /// Get a string representation of a coordinate handled by this axis.
  ///
  /// @param value Value to be converted to string
  /// @return a string representation of the value
  [[nodiscard]] virtual inline auto coordinate_repr(const T value) const
      -> std::string {
    return std::to_string(value);
  }

 protected:
  /// Gets the axis handler
  ///
  /// @return the axis handler
  [[nodiscard]] inline auto handler() const noexcept
      -> const std::shared_ptr<axis::container::Abstract<T>> & {
    return axis_;
  }

  /// Construction of a serialized instance.
  ///
  /// @param axis Axis handler
  /// @param is_circle True, if the axis can represent a circle.
  Axis(std::shared_ptr<axis::container::Abstract<T>> axis, const bool is_circle)
      : is_circle_(is_circle),
        circle_(is_circle_ ? T(360) : math::Fill<T>::value()),
        axis_(std::move(axis)) {}

 private:
  /// True, if the axis represents a circle.
  bool is_circle_{false};

  /// The value of the circle (360, π)
  T circle_{math::Fill<T>::value()};

  /// The object that handles access and searches for the values defined by
  /// the axis.
  std::shared_ptr<axis::container::Abstract<T>> axis_{
      std::make_shared<axis::container::Undefined<T>>()};

  /// Computes axis's properties
  void compute_properties(T epsilon) {
    // An axis can be represented by an empty set of values
    if (axis_->size() == 0) {
      throw std::invalid_argument("unable to create an empty axis.");
    }
    // Axis values must be sorted.
    if (!axis_->is_monotonic()) {
      throw std::invalid_argument("axis values are not ordered");
    }
    // If this axis represents an angle, determine if it represents the entire
    // trigonometric circle.
    if (is_angle()) {
      auto ptr = dynamic_cast<axis::container::Regular<T> *>(axis_.get());
      if (ptr != nullptr) {
        is_circle_ = math::is_same(
            static_cast<T>(std::fabs(ptr->step() * size())), circle_, epsilon);
      } else {
        auto increment = (axis_->back() - axis_->front()) /
                         static_cast<T>(axis_->size() - 1);
        is_circle_ = std::fabs((axis_->max_value() - axis_->min_value()) -
                               circle_) <= increment;
      }
    }
  }

  /// Put longitude into the range [0, circle_] degrees.
  auto normalize_longitude(const Vector<T> &points)
      -> std::unique_ptr<Vector<T>> {
    auto monotonic = true;
    auto ascending = points.size() < 2 ? true : points[0] < points[1];

    for (Eigen::Index ix = 1; ix < points.size(); ++ix) {
      monotonic =
          ascending ? points[ix - 1] < points[ix] : points[ix - 1] > points[ix];

      if (!monotonic) {
        break;
      }
    }

    if (!monotonic) {
      auto result = std::make_unique<Vector<T>>(points);
      auto cross = false;

      for (Eigen::Index ix = 1; ix < result->size(); ++ix) {
        if (!cross) {
          cross = ascending ? (*result)[ix - 1] > (*result)[ix]
                            : (*result)[ix - 1] < (*result)[ix];
        }

        if (cross) {
          (*result)[ix] += ascending ? circle_ : -circle_;
        }
      }
      return result;
    }
    return nullptr;
  }

  /// Initializes the axis container from values.
  auto initialize_from_values(const Eigen::Ref<const Vector<T>> &values,
                              const T epsilon, const bool move) -> void {
    // Determines whether the set of data provided can be represented as an
    // interval.
    auto increment = Axis<T>::is_evenly_spaced(values, epsilon);
    if (increment) {
      axis_ = std::make_shared<axis::container::Regular<T>>(
          values[0], values[values.size() - 1], static_cast<T>(values.size()));
    } else {
      // Avoid data copy if possible.
      axis_ = std::make_shared<axis::container::Irregular<T>>(
          move ? std::move(values) : values);
    }
    compute_properties(epsilon);
  }

  /// Determines whether the values contained in the vector are evenly spaced
  /// from each other.
  ///
  /// @param points Values to be tested
  /// @param epsilon Maximum allowed difference between two numbers in
  /// order to consider them equal
  /// @return The increment between two values if the values are evenly spaced
  static auto is_evenly_spaced(const Eigen::Ref<const Vector<T>> &points,
                               const T epsilon) -> std::optional<T> {
    size_t n = points.size();

    // The axis is defined by a single value.
    if (n < 2) {
      return {};
    }

    T increment =
        (points[points.size() - 1] - points[0]) / static_cast<T>(n - 1);

    // If the first two values are constant, the values are not evenly spaced.
    if (std::abs(increment) <= epsilon) {
      return {};
    }

    for (size_t ix = 1; ix < n; ++ix) {
      if (!math::is_same(points[ix] - points[ix - 1], increment, epsilon)) {
        return {};
      }
    }
    return increment;
  }
};

}  // namespace pyinterp::detail
