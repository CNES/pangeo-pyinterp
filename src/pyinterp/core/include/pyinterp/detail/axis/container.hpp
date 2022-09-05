// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

#include "pyinterp/detail/math.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::axis::container {

/// Abstraction of a container of values representing a mathematical axis.
///
/// @tparam T type of data handled by this container
template <typename T>
class Abstract {
 public:
  /// Default constructor
  Abstract() = default;

  /// Default destructor
  virtual ~Abstract() = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Abstract(const Abstract &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Abstract(Abstract &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Abstract &rhs) -> Abstract & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Abstract &&rhs) noexcept -> Abstract & = default;

  /// Returns true if the data is arranged in ascending order.
  [[nodiscard]] constexpr auto is_ascending() const -> bool {
    return is_ascending_;
  }

  /// Reverse the order of elements in this axis
  virtual auto flip() -> void = 0;

  /// Checks that the axis is monotonic
  [[nodiscard]] virtual auto is_monotonic() const -> bool { return true; }

  /// Get the ith coordinate value.
  ///
  /// @param index which coordinate. Between 0 and size()-1 inclusive
  /// @return coordinate value
  [[nodiscard]] virtual auto coordinate_value(int64_t index) const -> T = 0;

  /// Get a slice of the axis.
  ///
  /// @param start index of the first element to include in the slice
  /// @param count number of elements to include in the slice
  /// @return a slice of the axis
  [[nodiscard]] virtual auto slice(int64_t start, int64_t count) const
      -> Vector<T> = 0;

  /// Get the minimum coordinate value.
  ///
  /// @return minimum coordinate value
  [[nodiscard]] virtual auto min_value() const -> T = 0;

  /// Get the maximum coordinate value.
  ///
  /// @return maximum coordinate value
  [[nodiscard]] virtual auto max_value() const -> T = 0;

  /// Get the number of values for this axis
  ///
  /// @return the number of values
  [[nodiscard]] virtual auto size() const -> int64_t = 0;

  /// Gets the first element in the container
  ///
  /// @return the first element
  [[nodiscard]] virtual auto front() const -> T = 0;

  /// Gets the last element in the container
  ///
  /// @return the last element
  [[nodiscard]] virtual auto back() const -> T = 0;

  /// Search for the index corresponding to the requested value.
  ///
  /// @param coordinate position in this coordinate system
  /// @param bounded if true, returns "-1" if the value is located outside this
  /// coordinate system, otherwise the value of the first element if the value
  /// is located before, or the value of the last element of this container if
  /// the requested value is located after.
  /// @return index of the requested value it or -1 if outside this coordinate
  /// system area.
  [[nodiscard]] virtual auto find_index(T coordinate, bool bounded) const
      -> int64_t = 0;

  /// compare two variables instances
  ///
  /// @param rhs A variable to compare
  /// @return if variables are equals
  virtual auto operator==(const Abstract &rhs) const -> bool = 0;

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
  /// @param rhs right value
  Undefined(const Undefined &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Undefined(Undefined &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Undefined &rhs) -> Undefined & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Undefined &&rhs) noexcept -> Undefined & = default;

  /// @copydoc Abstract::flip()
  auto flip() -> void override {}

  /// @copydoc Abstract::coordinate_value(const int64_t) const
  [[nodiscard]] constexpr auto coordinate_value(
      const int64_t /* index */) const noexcept -> T override {
    return math::Fill<T>::value();
  }

  /// @copydoc Abstract::slice(const int64_t, const int64_t) const
  [[nodiscard]] constexpr auto slice(const int64_t /* start */,
                                     const int64_t /* count */) const noexcept
      -> Vector<T> override {
    auto result = Vector<T>(1);
    result[0] = math::Fill<T>::value();
    return result;
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

  /// @copydoc Abstract::find_index(double,bool) const
  constexpr auto find_index(T /* coordinate */,
                            bool /* bounded */) const  /// NOLINT
      noexcept -> int64_t override {
    return -1;
  }

  /// @copydoc Abstract::operator==(const Abstract&) const
  constexpr auto operator==(const Abstract<T> &rhs) const noexcept
      -> bool override {
    return dynamic_cast<const Undefined<T> *>(&rhs) != nullptr;
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
  /// @param points axis values
  explicit Irregular(Vector<T> points) : points_(std::move(points)) {
    if (points_.size() == 0) {
      throw std::invalid_argument("unable to create an empty container.");
    }
    this->is_ascending_ = this->calculate_is_ascending();
  }

  /// Destructor
  ~Irregular() override = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Irregular(const Irregular &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Irregular(Irregular &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Irregular &rhs) -> Irregular & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Irregular &&rhs) noexcept -> Irregular & = default;

  /// @copydoc Abstract::flip()
  auto flip() -> void override {
    std::reverse(points_.data(), points_.data() + points_.size());
    this->is_ascending_ = !this->is_ascending_;
  }

  /// @copydoc Abstract::is_monotonic() const
  [[nodiscard]] inline auto is_monotonic() const noexcept -> bool override {
    auto start = points_.data();
    auto end = points_.data() + points_.size();
    if (std::adjacent_find(start, end) != end) {
      return false;
    }
    if (this->is_ascending_) {
      return std::is_sorted(start, end);
    }
    return std::is_sorted(start, end, std::greater<>());
  };

  /// @copydoc Abstract::coordinate_value(const int64_t) const
  [[nodiscard]] constexpr auto coordinate_value(const int64_t index) const
      -> T override {
    return points_[index];
  }

  /// @copydoc Abstract::slice(const int64_t, const int64_t) const
  [[nodiscard]] constexpr auto slice(const int64_t start,
                                     const int64_t count) const noexcept
      -> Vector<T> override {
    return points_.segment(start, count);
  }

  /// @copydoc Abstract::min_value() const
  [[nodiscard]] constexpr auto min_value() const -> T override {
    return this->is_ascending_ ? front() : back();
  }

  /// @copydoc Abstract::max_value() const
  [[nodiscard]] constexpr auto max_value() const -> T override {
    return this->is_ascending_ ? back() : front();
  }

  /// @copydoc Abstract::size() const
  [[nodiscard]] constexpr auto size() const noexcept -> int64_t override {
    return points_.size();
  }

  /// @copydoc Abstract::front() const
  [[nodiscard]] constexpr auto front() const -> T override {
    return points_[0];
  }

  /// @copydoc Abstract::back() const
  [[nodiscard]] constexpr auto back() const -> T override {
    return points_[points_.size() - 1];
  }

  /// @copydoc Abstract::find_index(double,bool) const
  [[nodiscard]] constexpr auto find_index(const T coordinate,
                                          const bool bounded) const
      -> int64_t override {
    if (this->is_ascending_) {
      return this->find_index(coordinate, bounded, size(), std::less<T>());
    }
    return this->find_index(coordinate, bounded, size(), std::greater<T>());
  }

  /// @copydoc Abstract::operator==(const Abstract&) const
  auto operator==(const Abstract<T> &rhs) const noexcept -> bool override {
    const auto ptr = dynamic_cast<const Irregular<T> *>(&rhs);
    if (ptr != nullptr) {
      return ptr->points_.size() == points_.size() && ptr->points_ == points_;
    }
    return false;
  }

 private:
  Vector<T> points_{};

  /// Search for the index corresponding to the requested value if the axis is
  /// sorted in ascending order.
  template <typename Compare>
  [[nodiscard]] constexpr auto find_index(const T coordinate,
                                          const bool bounded, int64_t size,
                                          Compare cmp) const -> int64_t {
    auto begin = points_.data();
    auto end = points_.data() + size;
    auto it = std::lower_bound(begin, end, coordinate, cmp);

    if (it == begin) {
      if (cmp(coordinate, *it)) {
        return bounded ? 0 : -1;
      }
      return 0;
    }

    if (it == end) {
      if (cmp(*(it - 1), coordinate)) {
        return bounded ? size - 1 : -1;
      }
      return size - 1;
    }

    if (abs(coordinate - *(it - 1)) < abs(coordinate - *it)) {
      it--;
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
  /// @param start the starting value of the sequence
  /// @param stop the end value of the sequence
  /// @param num number of samples in the container
  AbstractRegular(const T start, const T stop, const T num)
      : size_(static_cast<int64_t>(num)), start_(start) {
    if (num == 0) {
      throw std::invalid_argument("unable to create an empty container.");
    }
    step_ = num == 1 ? stop - start : (stop - start) / (num - 1);
    // The inverse step of this axis is stored in order to optimize the search
    // for an index for a given value by avoiding a division.
    this->is_ascending_ = this->calculate_is_ascending();
  }

  /// Destructor
  ~AbstractRegular() override = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  AbstractRegular(const AbstractRegular &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  AbstractRegular(AbstractRegular &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const AbstractRegular &rhs) -> AbstractRegular & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(AbstractRegular &&rhs) noexcept -> AbstractRegular & = default;

  /// Get the step between two successive values.
  ///
  /// @return increment value
  [[nodiscard]] auto constexpr step() const -> T { return step_; }

  /// @copydoc Abstract::flip()
  auto flip() -> void override {
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
    auto result = Vector<T>(count);
    for (int64_t ix = 0; ix < count; ++ix) {
      result[ix] = coordinate_value(start + ix);
    }
    return result;
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
    const auto ptr = dynamic_cast<const AbstractRegular<T> *>(&rhs);
    if (ptr != nullptr) {
      return ptr->step_ == step_ && ptr->start_ == start_ &&
             ptr->size_ == size_;
    }
    return false;
  }

 protected:
  /// Container size.
  int64_t size_{};
  /// Value of the first item in the container.
  T start_{};
  /// The step between two succeeding values.
  T step_{};
};

/// Represents a container for an regularly spaced axis
///
template <typename T, class Enable = void>
class Regular : public AbstractRegular<T> {
 public:
  using AbstractRegular<T>::AbstractRegular;
};

/// Represents a container for an regularly spaced axis
///
template <typename T>
class Regular<T, typename std::enable_if<std::is_floating_point_v<T>>::type>
    : public AbstractRegular<T> {
 public:
  /// @copydoc AbstractRegular::AbstractRegular(const T start, const T stop,
  /// const T num)
  Regular(const T start, const T stop, const T num)
      // The inverse step of this axis is stored in order to optimize the search
      // for an index for a given value by avoiding a division.
      : AbstractRegular<T>(start, stop, num), inv_step_(T(1.0) / this->step_) {}

  /// @copydoc Abstract::find_index(double,bool) const
  [[nodiscard]] auto find_index(T coordinate, bool bounded) const noexcept
      -> int64_t override {
    auto index = static_cast<int64_t>(
        std::round((coordinate - this->start_) * inv_step_));

    if (index < 0) {
      return bounded ? 0 : -1;
    }

    if (index >= this->size_) {
      return bounded ? this->size_ - 1 : -1;
    }
    return index;
  }

  /// @copydoc Abstract::flip()
  auto flip() -> void override {
    AbstractRegular<T>::flip();
    inv_step_ = -inv_step_;
  }

 private:
  /// The inverse of the step (to avoid a division between real numbers).
  T inv_step_{};
};

/// Represents a container for an regularly spaced axis
///
template <typename T>
class Regular<T, typename std::enable_if<std::is_integral_v<T>>::type>
    : public AbstractRegular<T> {
 public:
  /// @copydoc AbstractRegular::AbstractRegular(const T start, const T stop,
  /// const T num)
  Regular(const T start, const T stop, const T num)
      // The inverse step of this axis is stored in order to optimize the search
      // for an index for a given value by avoiding a division.
      : AbstractRegular<T>(start, stop, num), step_2_(this->step_ >> 1) {}

  /// @copydoc Abstract::find_index(double,bool) const
  [[nodiscard]] constexpr auto find_index(T coordinate,
                                          bool bounded) const noexcept
      -> int64_t override {
    auto index =
        static_cast<int64_t>(round((coordinate - this->start_), this->step_));

    if (index < 0) {
      return bounded ? 0 : -1;
    }

    if (index >= this->size_) {
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

}  // namespace pyinterp::detail::axis::container
