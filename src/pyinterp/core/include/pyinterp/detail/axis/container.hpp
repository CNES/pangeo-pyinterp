// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pyinterp::detail::axis::container {

/// Abstraction of a container of values representing a mathematical axis.
class Abstract {
 public:
  /// Default constructor
  Abstract() = default;

  /// Default destructor
  virtual ~Abstract() = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Abstract(const Abstract& rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Abstract(Abstract&& rhs) = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Abstract& rhs) -> Abstract& = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Abstract&& rhs) -> Abstract& = default;

  /// Returns true if the data is arranged in ascending order.
  [[nodiscard]] inline auto is_ascending() const -> bool {
    return is_ascending_;
  }

  /// Reverse the order of elements in this axis
  virtual auto flip() -> void = 0;

  /// Checks thats axis is monotonic
  [[nodiscard]] virtual auto is_monotonic() const -> bool { return true; }

  /// Get the ith coordinate value.
  ///
  /// @param index which coordinate. Between 0 and size()-1 inclusive
  /// @return coordinate value
  [[nodiscard]] virtual auto coordinate_value(size_t index) const -> double = 0;

  /// Get the minimum coordinate value.
  ///
  /// @return minimum coordinate value
  [[nodiscard]] virtual auto min_value() const -> double = 0;

  /// Get the maximum coordinate value.
  ///
  /// @return maximum coordinate value
  [[nodiscard]] virtual auto max_value() const -> double = 0;

  /// Get the number of values for this axis
  ///
  /// @return the number of values
  [[nodiscard]] virtual auto size() const -> int64_t = 0;

  /// Gets the first element in the container
  ///
  /// @return the first element
  [[nodiscard]] virtual auto front() const -> double = 0;

  /// Gets the last element in the container
  ///
  /// @return the last element
  [[nodiscard]] virtual auto back() const -> double = 0;

  /// Search for the index corresponding to the requested value.
  ///
  /// @param coordinate position in this coordinate system
  /// @param bounded if true, returns "-1" if the value is located outside this
  /// coordinate system, otherwise the value of the first element if the value
  /// is located before, or the value of the last element of this container if
  /// the requested value is located after.
  /// @return index of the requested value it or -1 if outside this coordinate
  /// system area.
  [[nodiscard]] virtual auto find_index(double coordinate, bool bounded) const
      -> int64_t = 0;

  /// compare two variables instances
  ///
  /// @param rhs A variable to compare
  /// @return if variables are equals
  virtual auto operator==(const Abstract& rhs) const -> bool = 0;

 protected:
  /// Indicates whether the data is stored in the ascending order.
  bool is_ascending_{true};

  /// Calculate if the data is arranged in ascending order.
  [[nodiscard]] inline auto calculate_is_ascending() const -> bool {
    return size() < 2 ? true : coordinate_value(0) < coordinate_value(1);
  }
};

/// Represents a container for an undefined axis
class Undefined : public Abstract {
 public:
  /// Default constructor
  Undefined() = default;

  /// Default destructor
  ~Undefined() override = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Undefined(const Undefined& rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Undefined(Undefined&& rhs) = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Undefined& rhs) -> Undefined& = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Undefined&& rhs) -> Undefined& = default;

  /// @copydoc Abstract::flip()
  auto flip() -> void override {}

  /// @copydoc Abstract::coordinate_value(const size_t) const
  [[nodiscard]] inline auto coordinate_value(const size_t /* index */) const
      noexcept -> double override {
    return std::numeric_limits<double>::quiet_NaN();
  }

  /// @copydoc Abstract::min_value() const
  [[nodiscard]] inline auto min_value() const noexcept -> double override {
    return coordinate_value(0);
  }

  /// @copydoc Abstract::max_value() const
  [[nodiscard]] inline auto max_value() const noexcept -> double override {
    return coordinate_value(0);
  }

  /// @copydoc Abstract::size() const
  [[nodiscard]] inline auto size() const noexcept -> int64_t override {
    return 0;
  }

  /// @copydoc Abstract::front() const
  [[nodiscard]] inline auto front() const noexcept -> double override {
    return coordinate_value(0);
  }

  /// @copydoc Abstract::back() const
  [[nodiscard]] inline auto back() const noexcept -> double override {
    return coordinate_value(0);
  }

  /// @copydoc Abstract::find_index(double,bool) const
  inline int64_t find_index(double coordinate, bool bounded) const  /// NOLINT
      noexcept override {
    return -1;
  }

  /// @copydoc Abstract::operator==(const Abstract&) const
  inline auto operator==(const Abstract& rhs) const noexcept -> bool override {
    return dynamic_cast<const Undefined*>(&rhs) != nullptr;
  }
};

/// Represents a container for an irregularly spaced axis
class Irregular : public Abstract {
 public:
  /// Creation of a container representing an irregularly spaced coordinate
  /// system.
  ///
  /// @param points axis values
  explicit Irregular(Eigen::VectorXd points);

  /// Destructor
  ~Irregular() override = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Irregular(const Irregular& rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Irregular(Irregular&& rhs) = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Irregular& rhs) -> Irregular& = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Irregular&& rhs) -> Irregular& = default;

  /// @copydoc Abstract::flip()
  auto flip() -> void override;

  /// @copydoc Abstract::is_monotonic() const
  [[nodiscard]] inline auto is_monotonic() const noexcept -> bool override {
    if (is_ascending_) {
      return std::is_sorted(points_.data(), points_.data() + points_.size());
    }
    return std::is_sorted(points_.data(), points_.data() + points_.size(),
                          std::greater<>());
  };

  /// @copydoc Abstract::coordinate_value(const size_t) const
  [[nodiscard]] inline auto coordinate_value(const size_t index) const
      -> double override {
    return points_[index];
  }

  /// @copydoc Abstract::min_value() const
  [[nodiscard]] inline auto min_value() const -> double override {
    return is_ascending_ ? front() : back();
  }

  /// @copydoc Abstract::max_value() const
  [[nodiscard]] inline auto max_value() const -> double override {
    return is_ascending_ ? back() : front();
  }

  /// @copydoc Abstract::size() const
  [[nodiscard]] inline auto size() const noexcept -> int64_t override {
    return points_.size();
  }

  /// @copydoc Abstract::front() const
  [[nodiscard]] inline auto front() const -> double override {
    return points_[0];
  }

  /// @copydoc Abstract::back() const
  [[nodiscard]] inline auto back() const -> double override {
    return points_[points_.size() - 1];
  }

  /// @copydoc Abstract::find_index(double,bool) const
  [[nodiscard]] auto find_index(double coordinate, bool bounded) const
      -> int64_t override;

  /// @copydoc Abstract::operator==(const Abstract&) const
  auto operator==(const Abstract& rhs) const noexcept -> bool override {
    const auto ptr = dynamic_cast<const Irregular*>(&rhs);
    if (ptr != nullptr) {
      return ptr->points_.size() == points_.size() && ptr->points_ == points_;
    }
    return false;
  }

 private:
  Eigen::VectorXd points_{};
  Eigen::VectorXd edges_{};

  /// Computes the edges, if the axis data are not spaced regularly.
  void make_edges();
};

/// Represents a container for an regularly spaced axis
class Regular : public Abstract {
 public:
  /// Create a container from evenly spaced numbers over a specified
  /// interval.
  ///
  /// @param start the starting value of the sequence
  /// @param stop the end value of the sequence
  /// @param num number of samples in the container
  Regular(const double start, const double stop, const double num)
      : size_(static_cast<int64_t>(num)), start_(start) {
    if (num == 0) {
      throw std::invalid_argument("unable to create an empty container.");
    }
    step_ = num == 1 ? stop - start : (stop - start) / (num - 1);
    // The inverse step of this axis is stored in order to optimize the search
    // for an index for a given value by avoiding a division.
    inv_step_ = 1.0 / step_;
    is_ascending_ = calculate_is_ascending();
  }

  /// Destructor
  ~Regular() override = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Regular(const Regular& rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Regular(Regular&& rhs) = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Regular& rhs) -> Regular& = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Regular&& rhs) -> Regular& = default;

  /// Get the step between two successive values.
  ///
  /// @return increment value
  [[nodiscard]] auto step() const -> double { return step_; }

  /// @copydoc Abstract::flip()
  auto flip() -> void override {
    start_ = back();
    step_ = -step_;
    inv_step_ = -inv_step_;
    is_ascending_ = !is_ascending_;
  }

  /// @copydoc Abstract::coordinate_value(const size_t) const
  [[nodiscard]] inline auto coordinate_value(const size_t index) const noexcept
      -> double override {
    return start_ + index * step_;
  }

  /// @copydoc Abstract::find_index(double,bool) const
  [[nodiscard]] auto find_index(double coordinate, bool bounded) const noexcept
      -> int64_t override {
    auto index =
        static_cast<int64_t>(std::round((coordinate - start_) * inv_step_));

    if (index < 0) {
      return bounded ? 0 : -1;
    }
    if (index >= size_) {
      return bounded ? size_ - 1 : -1;
    }
    return index;
  }

  /// @copydoc Abstract::min_value() const
  [[nodiscard]] inline auto min_value() const noexcept -> double override {
    return coordinate_value(is_ascending_ ? 0 : size_ - 1);
  }

  /// @copydoc Abstract::max_value() const
  [[nodiscard]] inline auto max_value() const noexcept -> double override {
    return coordinate_value(is_ascending_ ? size_ - 1 : 0);
  }

  /// @copydoc Abstract::front() const
  [[nodiscard]] inline auto front() const noexcept -> double override {
    return start_;
  }

  /// @copydoc Abstract::back() const
  [[nodiscard]] inline auto back() const noexcept -> double override {
    return coordinate_value(size_ - 1);
  }

  /// @copydoc Abstract::size() const
  [[nodiscard]] inline auto size() const noexcept -> int64_t override {
    return size_;
  }

  /// @copydoc Abstract::operator==(const Abstract&) const
  auto operator==(const Abstract& rhs) const noexcept -> bool override {
    const auto ptr = dynamic_cast<const Regular*>(&rhs);
    if (ptr != nullptr) {
      return ptr->step_ == step_ && ptr->start_ == start_ &&
             ptr->size_ == size_;
    }
    return false;
  }

 private:
  /// Container size.
  int64_t size_{};
  /// Value of the first item in the container.
  double start_{};
  /// The step between two succeeding values.
  double step_{};
  /// The inverse of the step (to avoid a division between real numbers).
  double inv_step_{};
};

}  // namespace pyinterp::detail::axis::container
