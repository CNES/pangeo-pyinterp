// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"

namespace pyinterp::math::axis {

/// Type of axis container
enum class AxisType : uint8_t {
  kUndefined,  //!< Undefined axis type
  kRegular,    //!< Regularly spaced axis
  kIrregular,  //!< Irregularly spaced axis
};

/// Abstraction of a container of values representing a mathematical axis.
///
/// @tparam T type of data handled by this container
template <typename T>
  requires std::is_arithmetic_v<T>
class Abstract {
 public:
  /// Default constructor
  Abstract() = default;

  /// Default destructor
  virtual ~Abstract() = default;

  /// Copy constructor
  ///
  /// @param[in] rhs right value
  Abstract(const Abstract &rhs) = default;

  /// Move constructor
  ///
  /// @param[in,out] rhs right value
  Abstract(Abstract &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param[in] rhs right value
  auto operator=(const Abstract &rhs) -> Abstract & = default;

  /// Move assignment operator
  ///
  /// @param[in,out] rhs right value
  auto operator=(Abstract &&rhs) noexcept -> Abstract & = default;

  /// Returns true if the data is arranged in ascending order.
  [[nodiscard]] constexpr auto is_ascending() const noexcept -> bool {
    return is_ascending_;
  }

  /// Get the type of this axis container
  [[nodiscard]] virtual constexpr auto type() const noexcept -> AxisType = 0;

  /// Reverse the order of elements in this axis
  virtual auto flip() noexcept -> void = 0;

  /// Checks that the axis is monotonic
  [[nodiscard]] virtual auto is_monotonic() const noexcept -> bool {
    return true;
  }

  /// Get the ith coordinate value.
  ///
  /// @param[in] index which coordinate. Between 0 and size()-1 inclusive
  /// @return coordinate value
  [[nodiscard]] virtual auto coordinate_value(int64_t index) const -> T = 0;

  /// Get a slice of the axis.
  ///
  /// @param[in] start index of the first element to include in the slice
  /// @param[in] count number of elements to include in the slice
  /// @return a slice of the axis
  [[nodiscard]] virtual auto slice(int64_t start, int64_t count) const
      -> Vector<T> = 0;

  /// Get the minimum coordinate value.
  ///
  /// @return minimum coordinate value
  [[nodiscard]] virtual auto min_value() const noexcept -> T = 0;

  /// Get the maximum coordinate value.
  ///
  /// @return maximum coordinate value
  [[nodiscard]] virtual auto max_value() const noexcept -> T = 0;

  /// Get the number of values for this axis
  ///
  /// @return the number of values
  [[nodiscard]] virtual auto size() const -> int64_t = 0;

  /// Gets the first element in the container
  ///
  /// @return the first element
  [[nodiscard]] virtual auto front() const noexcept -> T = 0;

  /// Gets the last element in the container
  ///
  /// @return the last element
  [[nodiscard]] virtual auto back() const noexcept -> T = 0;

  /// Search for the index corresponding to the requested value.
  ///
  /// @param[in] coordinate position in this coordinate system
  /// @param[in] bounded if false, returns "-1" if the value is located outside
  /// this coordinate system, otherwise the value of the first element if the
  /// value is located before, or the value of the last element of this
  /// container if the requested value is located after.
  /// @return index of the requested value it or -1 if outside this coordinate
  /// system area.
  [[nodiscard]] virtual auto find_index(T coordinate, bool bounded) const
      -> int64_t = 0;

  /// compare two variables instances
  ///
  /// @param[in] rhs A variable to compare
  /// @return if variables are equals
  virtual constexpr auto operator==(const Abstract<T> &rhs) const noexcept
      -> bool {
    // Use typeid instead of dynamic_cast for constexpr compatibility
    return typeid(*this) == typeid(rhs);
  }

  /// compare two variables instances
  ///
  /// @param[in] rhs A variable to compare
  /// @return if variables are not equals
  constexpr auto operator!=(const Abstract &rhs) const -> bool {
    return !(*this == rhs);
  }

 protected:
  /// Indicates whether the data is stored in the ascending order.
  bool is_ascending_{true};

  /// Calculate if the data is arranged in ascending order.
  [[nodiscard]] constexpr auto calculate_is_ascending() const -> bool {
    return size() < 2 ? true : coordinate_value(0) < coordinate_value(1);
  }
};

/// Represents a container for an undefined axis
///
/// @tparam T type of data handled by this container
template <typename T>
class Undefined : public Abstract<T> {
 public:
  /// Default constructor
  Undefined() = default;

  /// Default destructor
  ~Undefined() override = default;

  /// Copy constructor
  ///
  /// @param[in] rhs right value
  Undefined(const Undefined &rhs) = default;

  /// Move constructor
  ///
  /// @param[in,out] rhs right value
  Undefined(Undefined &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param[in] rhs right value
  auto operator=(const Undefined &rhs) -> Undefined & = default;

  /// Move assignment operator
  ///
  /// @param[in,out] rhs right value
  auto operator=(Undefined &&rhs) noexcept -> Undefined & = default;

  /// @copydoc Abstract::type()
  [[nodiscard]] constexpr auto type() const noexcept -> AxisType override {
    return AxisType::kUndefined;
  }

  /// @copydoc Abstract::flip()
  auto flip() noexcept -> void override {}

  /// @brief Get the ith coordinate value.
  /// @param[in] index which coordinate. Between 0 and size()-1 inclusive
  /// @return coordinate value
  [[nodiscard]] constexpr auto coordinate_value(
      const int64_t /* index */) const noexcept -> T override {
    return math::Fill<T>::value();
  }

  /// @brief Get a slice of the axis.
  /// @param[in] start index of the first element to include in the slice
  /// @param[in] count number of elements to include in the slice
  /// @return a slice of the axis
  [[nodiscard]] constexpr auto slice(const int64_t /* start */,
                                     const int64_t /* count */) const noexcept
      -> Vector<T> override {
    return Eigen::Vector<T, 1>{math::Fill<T>::value()};
  }

  /// @copydoc Abstract::min_value() const
  [[nodiscard]] constexpr auto min_value() const noexcept -> T override {
    return coordinate_value(0);
  }

  /// @copydoc Abstract::max_value() const
  [[nodiscard]] constexpr auto max_value() const noexcept -> T override {
    return coordinate_value(0);
  }

  /// @copydoc Abstract::size() const
  [[nodiscard]] constexpr auto size() const noexcept -> int64_t override {
    return 0;
  }

  /// @copydoc Abstract::front() const
  [[nodiscard]] constexpr auto front() const noexcept -> T override {
    return coordinate_value(0);
  }

  /// @copydoc Abstract::back() const
  [[nodiscard]] constexpr auto back() const noexcept -> T override {
    return coordinate_value(0);
  }

  /// @brief Search for the index corresponding to the requested value.
  /// @return Always returns -1
  [[nodiscard]]
  constexpr auto find_index(T /* coordinate */,
                            bool /* bounded */) const  /// NOLINT
      noexcept -> int64_t override {
    return -1;
  }
};

/// Represents a container for an irregularly spaced axis
///
/// @tparam T type of data handled by this container
template <typename T>
class Irregular : public Abstract<T> {
 public:
  /// Creation of a container representing an irregularly spaced coordinate
  /// system.
  ///
  /// @param[in] points axis values
  explicit Irregular(Vector<T> points) : points_(std::move(points)) {
    if (points_.size() == 0) {
      throw std::invalid_argument(
          "Cannot create an Irregular axis with no points.");
    }
    this->is_ascending_ = this->calculate_is_ascending();
  }

  /// Destructor
  ~Irregular() override = default;

  /// Copy constructor
  ///
  /// @param[in] rhs right value
  Irregular(const Irregular &rhs) = default;

  /// Move constructor
  ///
  /// @param[in,out] rhs right value
  Irregular(Irregular &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param[in] rhs right value
  auto operator=(const Irregular &rhs) -> Irregular & = default;

  /// Move assignment operator
  ///
  /// @param[in,out] rhs right value
  auto operator=(Irregular &&rhs) noexcept -> Irregular & = default;

  /// @copydoc Abstract::type()
  [[nodiscard]] constexpr auto type() const noexcept -> AxisType override {
    return AxisType::kIrregular;
  }

  /// @copydoc Abstract::flip()
  auto flip() noexcept -> void override {
    std::reverse(points_.begin(), points_.end());
    this->is_ascending_ = !this->is_ascending_;
  }

  /// @copydoc Abstract::is_monotonic() const
  [[nodiscard]] auto is_monotonic() const noexcept -> bool override {
    auto begin = points_.begin();
    auto end = points_.end();
    // Check for duplicates first (cheaper than full sort check)
    if (std::adjacent_find(begin, end) != end) {
      return false;
    }
    // Single pass check using ranges (C++20)
    return this->is_ascending_
               ? std::ranges::is_sorted(points_)
               : std::ranges::is_sorted(points_, std::greater{});
  }

  /// @copydoc Abstract::coordinate_value(const int64_t) const
  [[nodiscard]] constexpr auto coordinate_value(const int64_t index) const
      -> T override {
    return points_[index];
  }

  /// Get the underlying points of this axis.
  [[nodiscard]] constexpr auto points() const noexcept -> const Vector<T> & {
    return points_;
  }

  /// @copydoc Abstract::slice(const int64_t, const int64_t) const
  [[nodiscard]] constexpr auto slice(const int64_t start,
                                     const int64_t count) const noexcept
      -> Vector<T> override {
    return points_.segment(start, count);
  }

  /// @copydoc Abstract::min_value() const
  [[nodiscard]] constexpr auto min_value() const noexcept -> T override {
    return this->is_ascending_ ? front() : back();
  }

  /// @copydoc Abstract::max_value() const
  [[nodiscard]] constexpr auto max_value() const noexcept -> T override {
    return this->is_ascending_ ? back() : front();
  }

  /// @copydoc Abstract::size() const
  [[nodiscard]] constexpr auto size() const noexcept -> int64_t override {
    return points_.size();
  }

  /// @copydoc Abstract::front() const
  [[nodiscard]] constexpr auto front() const noexcept -> T override {
    return points_[0];
  }

  /// @copydoc Abstract::back() const
  [[nodiscard]] constexpr auto back() const noexcept -> T override {
    return points_[points_.size() - 1];
  }

  /// @copydoc Abstract::find_index(T,bool) const
  [[nodiscard]] constexpr auto find_index(const T coordinate,
                                          const bool bounded) const
      -> int64_t override {
    return this->is_ascending_
               ? find_index_impl(coordinate, bounded, size(), std::less{})
               : find_index_impl(coordinate, bounded, size(), std::greater{});
  }

  /// @copydoc Abstract::operator==(const Abstract&) const
  auto operator==(const Abstract<T> &rhs) const noexcept -> bool override {
    const auto *other = dynamic_cast<const Irregular<T> *>(&rhs);
    return other != nullptr && other->points_.size() == points_.size() &&
           std::ranges::equal(points_, other->points_);
  }

 private:
  /// Container points.
  Vector<T> points_{};

  /// Search for the index corresponding to the requested value.
  template <typename Compare>
  [[nodiscard]] constexpr auto find_index_impl(const T coordinate,
                                               const bool bounded,
                                               const int64_t size,
                                               Compare cmp) const -> int64_t {
    auto begin = points_.begin();
    auto end = points_.end();
    auto it = std::lower_bound(begin, end, coordinate, cmp);

    if (it == begin) {
      return cmp(coordinate, *it) ? (bounded ? 0 : -1) : 0;
    }

    if (it == end) {
      return cmp(*(end - 1), coordinate) ? (bounded ? size - 1 : -1) : size - 1;
    }

    const auto prev = it - 1;
    if (cmp((coordinate - *prev), (*it - coordinate))) {
      return std::distance(begin, prev);
    }
    return std::distance(begin, it);
  }
};

/// Represents a container for an regularly spaced axis
///
/// @tparam T type of data handled by this container
template <typename T>
class AbstractRegular : public Abstract<T> {
 public:
  /// Create a container from evenly spaced numbers over a specified
  /// interval.
  ///
  /// @param[in] start the starting value of the sequence
  /// @param[in] stop the end value of the sequence
  /// @param[in] num number of samples in the container
  AbstractRegular(const T start, const T stop, const size_t num)
      : size_(static_cast<int64_t>(num)), start_(start) {
    if (num == 0) {
      throw std::invalid_argument(
          "The number of samples must be greater than zero to create a valid "
          "axis.");
    }
    if (start == stop) {
      throw std::invalid_argument(
          "An axis with a single value requires distinct start and stop "
          "values.");
    }
    step_ = static_cast<T>(num == 1 ? stop - start
                                    : (stop - start) /
                                          static_cast<int64_t>(num - 1));
    // The inverse step of this axis is stored in order to optimize the search
    // for an index for a given value by avoiding a division.
    this->is_ascending_ = this->calculate_is_ascending();
  }

  /// Destructor
  ~AbstractRegular() override = default;

  /// Copy constructor
  ///
  /// @param[in] rhs right value
  AbstractRegular(const AbstractRegular &rhs) = default;

  /// Move constructor
  ///
  /// @param[in,out] rhs right value
  AbstractRegular(AbstractRegular &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param[in] rhs right value
  auto operator=(const AbstractRegular &rhs) -> AbstractRegular & = default;

  /// Move assignment operator
  ///
  /// @param[in,out] rhs right value
  auto operator=(AbstractRegular &&rhs) noexcept -> AbstractRegular & = default;

  /// Get the step between two successive values.
  ///
  /// @return increment value
  [[nodiscard]] auto constexpr step() const noexcept -> T { return step_; }

  /// @copydoc Abstract::type()
  [[nodiscard]] constexpr auto type() const noexcept -> AxisType override {
    return AxisType::kRegular;
  }

  /// @copydoc Abstract::flip()
  auto flip() noexcept -> void override {
    start_ = back();
    step_ = -step_;
    this->is_ascending_ = !this->is_ascending_;
  }

  /// @copydoc Abstract::coordinate_value(const int64_t) const
  [[nodiscard]] constexpr auto coordinate_value(
      const int64_t index) const noexcept -> T override {
    return static_cast<T>(start_ + index * step_);
  }

  /// @copydoc Abstract::slice(const int64_t, const int64_t) const
  [[nodiscard]] constexpr auto slice(const int64_t start,
                                     const int64_t count) const noexcept
      -> Vector<T> override {
    return Vector<T>::LinSpaced(count, coordinate_value(start),
                                coordinate_value(start + count - 1));
  }

  /// @copydoc Abstract::min_value() const
  [[nodiscard]] constexpr auto min_value() const noexcept -> T override {
    return coordinate_value(this->is_ascending_ ? 0 : size_ - 1);
  }

  /// @copydoc Abstract::max_value() const
  [[nodiscard]] constexpr auto max_value() const noexcept -> T override {
    return coordinate_value(this->is_ascending_ ? size_ - 1 : 0);
  }

  /// @copydoc Abstract::front() const
  [[nodiscard]] constexpr auto front() const noexcept -> T override {
    return start_;
  }

  /// @copydoc Abstract::back() const
  [[nodiscard]] constexpr auto back() const noexcept -> T override {
    return coordinate_value(size_ - 1);
  }

  /// @copydoc Abstract::size() const
  [[nodiscard]] constexpr auto size() const noexcept -> int64_t override {
    return size_;
  }

  /// @copydoc Abstract::operator==(const Abstract&) const
  auto operator==(const Abstract<T> &rhs) const noexcept -> bool override {
    const auto *other = dynamic_cast<const AbstractRegular<T> *>(&rhs);
    return other != nullptr && other->step_ == step_ &&
           other->start_ == start_ && other->size_ == size_;
  }

 protected:
  /// Container size.
  int64_t size_{};
  /// Value of the first item in the container.
  T start_{};
  /// The step between two succeeding values.
  T step_{};
};

/// Represents a container for a regularly spaced axis
template <typename T, typename Enable = void>
class Regular;

/// Represents a container for a regularly spaced axis (floating-point
/// specialization)
template <typename T>
class Regular<T, std::enable_if_t<std::floating_point<T>>>
    : public AbstractRegular<T> {
 public:
  /// @copydoc AbstractRegular::AbstractRegular(const T start, const T stop,
  /// const size_t num)
  Regular(const T start, const T stop, const size_t num)
      : AbstractRegular<T>(start, stop, num), inv_step_(T(1.0) / this->step_) {}

  /// @copydoc Abstract::find_index(T, bool) const
  [[nodiscard]] auto find_index(const T coordinate,
                                const bool bounded) const noexcept
      -> int64_t override {
    const auto index = static_cast<int64_t>(
        std::round((coordinate - this->start_) * inv_step_));

    if (index < 0) [[unlikely]] {
      return bounded ? 0 : -1;
    }

    if (index >= this->size_) [[unlikely]] {
      return bounded ? this->size_ - 1 : -1;
    }
    return index;
  }

  /// @copydoc Abstract::flip()
  auto flip() noexcept -> void override {
    AbstractRegular<T>::flip();
    inv_step_ = -inv_step_;
  }

 private:
  /// The inverse of the step (to avoid a division between real numbers).
  T inv_step_{};
};

/// Represents a container for a regularly spaced axis (integral specialization)
template <typename T>
class Regular<T, std::enable_if_t<std::integral<T>>>
    : public AbstractRegular<T> {
 public:
  /// @copydoc AbstractRegular::AbstractRegular(const T start, const T stop,
  /// const size_t num)
  Regular(const T start, const T stop, const size_t num)
      : AbstractRegular<T>(start, stop, num), step_2_(this->step_ >> 1) {}

  /// @copydoc Abstract::find_index(T, bool) const
  [[nodiscard]] constexpr auto find_index(const T coordinate,
                                          const bool bounded) const noexcept
      -> int64_t override {
    const auto index =
        static_cast<int64_t>(round((coordinate - this->start_), this->step_));

    if (index < 0) [[unlikely]] {
      return bounded ? 0 : -1;
    }

    if (index >= this->size_) [[unlikely]] {
      return bounded ? this->size_ - 1 : -1;
    }
    return index;
  }

 private:
  /// The absolute value of half the pitch of this axis. This value is used to
  /// search for the nearest index.
  T step_2_{};

  /// Divide positive or negative dividend by positive divisor and round to
  /// closest integer
  [[nodiscard]] constexpr auto round(int64_t numerator,
                                     int64_t denominator) const -> int64_t {
    return numerator > 0 ? (numerator + step_2_) / denominator
                         : (numerator - step_2_) / denominator;
  }
};

}  // namespace pyinterp::math::axis
