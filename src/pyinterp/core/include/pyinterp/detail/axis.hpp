// Copyright (c) 2019 CNES
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

namespace pyinterp::detail {

/// A coordinate axis is a Variable that specifies one of the coordinates
/// of a variable's values.
class Axis {
 public:
  /// Type of boundary handling.
  enum Boundary : uint8_t {
    kExpand,  //!< Expand the boundary as a constant.
    kWrap,    //!< Circular boundary conditions.
    kSym,     //!< Symmetrical boundary conditions.
    kUndef,   //!< Boundary violation is not defined.
  };

  /// Default constructor
  Axis() = default;

  /// Create a coordinate axis from evenly spaced numbers over a specified
  /// interval.
  ///
  /// @param start the first value of the axis
  /// @param stop the last value of the axis
  /// @param num number of samples in the axis
  /// @param epsilon Maximum allowed difference between two real numbers in
  /// order to consider them equal.
  /// @param is_circle True, if the axis can represent a circle.
  /// @param is_radian True, if the coordinate system is radian.
  Axis(const double start, const double stop, const double num,
       const double epsilon, const bool is_circle, const bool is_radian)
      : circle_(Axis::set_circle(is_circle, is_radian)),
        axis_(std::make_shared<axis::container::Regular>(
            axis::container::Regular(start, stop, num))) {
    compute_properties(epsilon);
  }

  /// Create a coordinate axis from values.
  ///
  /// @param values Axis values
  /// @param epsilon Maximum allowed difference between two real numbers in
  /// order to consider them equal.
  /// @param is_circle True, if the axis can represent a circle.
  /// @param is_radian True, if the coordinate system is radian.
  explicit Axis(Eigen::Ref<Eigen::VectorXd> values, double epsilon,
                bool is_circle, bool is_radian);

  /// Destructor
  ~Axis() = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Axis(const Axis& rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Axis(Axis&& rhs) = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Axis& rhs) -> Axis& = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Axis&& rhs) -> Axis& = default;

  /// Get the ith coordinate value.
  ///
  /// @param index which coordinate. Between 0 and size()-1 inclusive
  /// @return coordinate value
  /// @throw std::out_of_range if !(index < size()).
  [[nodiscard]] inline auto coordinate_value(const size_t index) const
      -> double {
    if (static_cast<int64_t>(index) >= size()) {
      throw std::out_of_range("axis index out of range");
    }
    return axis_->coordinate_value(index);
  }

  /// Get the minimum coordinate value.
  ///
  /// @return minimum coordinate value
  [[nodiscard]] inline auto min_value() const -> double {
    return axis_->min_value();
  }

  /// Get the maximum coordinate value.
  ///
  /// @return maximum coordinate value
  [[nodiscard]] inline auto max_value() const -> double {
    return axis_->max_value();
  }

  /// Get the number of values for this axis
  ///
  /// @return the number of values
  [[nodiscard]] inline auto size() const noexcept -> int64_t {
    return axis_->size();
  }

  /// Check if this axis values are spaced regularly
  [[nodiscard]] inline auto is_regular() const noexcept -> bool {
    return dynamic_cast<axis::container::Regular*>(axis_.get()) != nullptr;
  }

  /// Returns true if this axis represents a circle.
  [[nodiscard]] inline constexpr auto is_circle() const noexcept -> bool {
    return is_circle_;
  }

  /// Does the axis represent an angle?
  ///
  /// @return true if the axis represent an angle
  [[nodiscard]] inline auto is_angle() const noexcept -> bool {
    return !std::isnan(circle_);
  }

  /// Get the first value of this axis
  ///
  /// @return the first value
  [[nodiscard]] inline auto front() const -> double { return axis_->front(); }

  /// Get the last value of this axis
  ///
  /// @return the last value
  [[nodiscard]] inline auto back() const -> double { return axis_->back(); }

  /// Test if the data is sorted in ascending order.
  ///
  /// @return True if the data is sorted in ascending order.
  [[nodiscard]] inline auto is_ascending() const -> bool {
    return axis_->is_ascending();
  }

  /// Reverse the order of elements in this axis
  auto flip() -> void { axis_->flip(); }

  /// Get increment value if is_regular()
  ///
  /// @return increment value if is_regular()
  /// @throw std::logic_error if this instance does not represent a regular axis
  [[nodiscard]] inline auto increment() const -> double {
    auto ptr = dynamic_cast<axis::container::Regular*>(axis_.get());
    if (ptr == nullptr) {
      throw std::logic_error("this axis is not regular.");
    }
    return ptr->step();
  }

  /// compare two variables instances
  ///
  /// @param rhs an other axis to compare
  /// @return if axis are equals
  inline auto operator==(Axis const& rhs) const -> bool {
    return *axis_ == *rhs.axis_ && is_circle_ == rhs.is_circle_;
  }

  /// compare two variables instances
  ///
  /// @param rhs an other axis to compare
  /// @return if axis are equals
  inline auto operator!=(Axis const& rhs) const -> bool {
    return !this->operator==(rhs);
  }

  /// Get the ith coordinate value.
  ///
  /// @param index which coordinate. Between 0 and size()-1 inclusive
  /// @return coordinate value
  inline auto operator()(const size_t index) const noexcept -> double {
    return axis_->coordinate_value(index);
  }

  /// Returns the normalized value with respect to the axis definition. This
  /// means if the axis defines a circle, this method returns a value within the
  /// interval [font(), back()] otherwise it returns the value supplied.
  [[nodiscard]] inline auto normalize_coordinate(const double coordinate) const
      noexcept -> double {
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
  [[nodiscard]] inline auto find_index(const double coordinate,
                                       const bool bounded) const -> int64_t {
    return axis_->find_index(normalize_coordinate(coordinate), bounded);
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
  [[nodiscard]] auto find_indexes(double coordinate) const
      -> std::optional<std::tuple<int64_t, int64_t>>;

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
  [[nodiscard]] auto find_indexes(double coordinate, uint32_t size,
                                  Boundary boundary) const
      -> std::vector<int64_t>;

  /// Get a string representing this instance.
  ///
  /// @return a string holding the converted instance.
  explicit operator std::string() const;

 protected:
  /// Specifies if this instance handles a radian angle.
  ///
  /// @return true if this instance handles a radian angle.
  [[nodiscard]] inline auto is_radian() const noexcept -> bool {
    return circle_ == math::pi<double>();
  }

  /// Gets the axis handler
  ///
  /// @return the axis handler
  [[nodiscard]] inline auto handler() const noexcept
      -> const std::shared_ptr<axis::container::Abstract>& {
    return axis_;
  }

  /// Construction of a serialized instance.
  ///
  /// @param axis Axis handler
  /// @param is_circle True, if the axis can represent a circle.
  /// @param is_radian True, if the coordinate system is radian.
  Axis(std::shared_ptr<axis::container::Abstract> axis, const bool is_circle,
       const bool is_radian)
      : is_circle_(is_circle),
        circle_(is_circle_ ? (is_radian ? math::pi<double>() : 360)
                           : std::numeric_limits<double>::quiet_NaN()),
        axis_(std::move(axis)) {}

 private:
  /// True, if the axis represents a circle.
  bool is_circle_{false};

  /// The value of the circle (360, π)
  double circle_{std::numeric_limits<double>::quiet_NaN()};

  /// The object that handles access and searches for the values defined by the
  /// axis.
  std::shared_ptr<axis::container::Abstract> axis_{
      std::make_shared<axis::container::Undefined>()};

  /// Determines if the axis represents a circle.
  inline static constexpr auto set_circle(const bool is_circle,
                                          const bool is_radian) noexcept
      -> double {
    return is_circle ? (is_radian ? math::pi<double>() : 360)
                     : std::numeric_limits<double>::quiet_NaN();
  }

  /// Normalize angle
  [[nodiscard]] inline auto normalize_coordinate(const double coordinate,
                                                 const double min) const
      noexcept -> double {
    if (is_angle() && (coordinate >= min + circle_ || coordinate < min)) {
      return math::normalize_angle(coordinate, min, circle_);
    }
    return coordinate;
  }

  /// Computes axis's properties
  void compute_properties(double epsilon);

  /// Put longitude into the range [0, circle_] degrees.
  void normalize_longitude(Eigen::Ref<Eigen::VectorXd>& points);  // NOLINT
};

}  // namespace pyinterp::detail
