// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

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
  Abstract& operator=(const Abstract& rhs) = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  Abstract& operator=(Abstract&& rhs) = default;

  /// Returns true if the data is arranged in ascending order.
  [[nodiscard]] inline bool is_ascending() const { return is_ascending_; }

  /// Checks thats axis is monotonic
  [[nodiscard]] virtual bool is_monotonic() const { return true; }

  /// Get the ith coordinate value.
  ///
  /// @param index which coordinate. Between 0 and size()-1 inclusive
  /// @return coordinate value
  [[nodiscard]] virtual double coordinate_value(size_t index) const = 0;

  /// Get the minimum coordinate value.
  ///
  /// @return minimum coordinate value
  [[nodiscard]] virtual double min_value() const = 0;

  /// Get the maximum coordinate value.
  ///
  /// @return maximum coordinate value
  [[nodiscard]] virtual double max_value() const = 0;

  /// Get the number of values for this axis
  ///
  /// @return the number of values
  [[nodiscard]] virtual int64_t size() const = 0;

  /// Gets the first element in the container
  ///
  /// @return the first element
  [[nodiscard]] virtual double front() const = 0;

  /// Gets the last element in the container
  ///
  /// @return the last element
  [[nodiscard]] virtual double back() const = 0;

  /// Search for the index corresponding to the requested value.
  ///
  /// @param coordinate position in this coordinate system
  /// @param bounded if true, returns "-1" if the value is located outside this
  /// coordinate system, otherwise the value of the first element if the value
  /// is located before, or the value of the last element of this container if
  /// the requested value is located after.
  /// @return index of the requested value it or -1 if outside this coordinate
  /// system area.
  [[nodiscard]] virtual int64_t find_index(double coordinate,
                                           bool bounded) const = 0;

  /// compare two variables instances
  ///
  /// @param rhs A variable to compare
  /// @return if variables are equals
  virtual bool operator==(const Abstract& rhs) const = 0;

 protected:
  /// Indicates whether the data is stored in the ascending order.
  bool is_ascending_{true};

  /// Calculate if the data is arranged in ascending order.
  [[nodiscard]] inline bool calculate_is_ascending() const {
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
  Undefined& operator=(const Undefined& rhs) = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  Undefined& operator=(Undefined&& rhs) = default;

  /// @copydoc Abstract::coordinate_value(const size_t) const
  [[nodiscard]] inline double coordinate_value(const size_t /* index */) const
      noexcept override {
    return std::numeric_limits<double>::quiet_NaN();
  }

  /// @copydoc Abstract::min_value() const
  [[nodiscard]] inline double min_value() const noexcept override {
    return coordinate_value(0);
  }

  /// @copydoc Abstract::max_value() const
  [[nodiscard]] inline double max_value() const noexcept override {
    return coordinate_value(0);
  }

  /// @copydoc Abstract::size() const
  [[nodiscard]] inline int64_t size() const noexcept override { return 0; }

  /// @copydoc Abstract::front() const
  [[nodiscard]] inline double front() const noexcept override {
    return coordinate_value(0);
  }

  /// @copydoc Abstract::back() const
  [[nodiscard]] inline double back() const noexcept override {
    return coordinate_value(0);
  }

  /// @copydoc Abstract::find_index(double,bool) const
  inline int64_t find_index(double coordinate, bool bounded) const  /// NOLINT
      noexcept override {
    return -1;
  }

  /// @copydoc Abstract::operator==(const Abstract&) const
  inline bool operator==(const Abstract& rhs) const noexcept override {
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
  explicit Irregular(std::vector<double> points);

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
  Irregular& operator=(const Irregular& rhs) = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  Irregular& operator=(Irregular&& rhs) = default;

  /// @copydoc Abstract::is_monotonic() const
  [[nodiscard]] inline bool is_monotonic() const noexcept override {
    if (is_ascending_) {
      return std::is_sorted(points_.begin(), points_.end());
    }
    return std::is_sorted(points_.rbegin(), points_.rend());
  };

  /// @copydoc Abstract::coordinate_value(const size_t) const
  [[nodiscard]] inline double coordinate_value(
      const size_t index) const override {
    return points_[index];
  }

  /// @copydoc Abstract::min_value() const
  [[nodiscard]] inline double min_value() const override {
    return is_ascending_ ? front() : back();
  }

  /// @copydoc Abstract::max_value() const
  [[nodiscard]] inline double max_value() const override {
    return is_ascending_ ? back() : front();
  }

  /// @copydoc Abstract::size() const
  [[nodiscard]] inline int64_t size() const noexcept override {
    return points_.size();
  }

  /// @copydoc Abstract::front() const
  [[nodiscard]] inline double front() const override { return points_.front(); }

  /// @copydoc Abstract::back() const
  [[nodiscard]] inline double back() const override { return points_.back(); }

  /// @copydoc Abstract::find_index(double,bool) const
  [[nodiscard]] int64_t find_index(double coordinate,
                                   bool bounded) const override;

  /// @copydoc Abstract::operator==(const Abstract&) const
  bool operator==(const Abstract& rhs) const noexcept override {
    const auto ptr = dynamic_cast<const Irregular*>(&rhs);
    if (ptr != nullptr) {
      return ptr->points_ == points_;
    }
    return false;
  }

 private:
  std::vector<double> points_{};
  std::vector<double> edges_{};

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
  Regular& operator=(const Regular& rhs) = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  Regular& operator=(Regular&& rhs) = default;

  /// Get the step between two successive values.
  ///
  /// @return increment value
  [[nodiscard]] double step() const { return step_; }

  /// @copydoc Abstract::coordinate_value(const size_t) const
  [[nodiscard]] inline double coordinate_value(const size_t index) const
      noexcept override {
    return start_ + index * step_;
  }

  /// @copydoc Abstract::find_index(double,bool) const
  [[nodiscard]] int64_t find_index(double coordinate, bool bounded) const
      noexcept override {
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
  [[nodiscard]] inline double min_value() const noexcept override {
    return coordinate_value(is_ascending_ ? 0 : size_ - 1);
  }

  /// @copydoc Abstract::max_value() const
  [[nodiscard]] inline double max_value() const noexcept override {
    return coordinate_value(is_ascending_ ? size_ - 1 : 0);
  }

  /// @copydoc Abstract::front() const
  [[nodiscard]] inline double front() const noexcept override { return start_; }

  /// @copydoc Abstract::back() const
  [[nodiscard]] inline double back() const noexcept override {
    return coordinate_value(size_ - 1);
  }

  /// @copydoc Abstract::size() const
  [[nodiscard]] inline int64_t size() const noexcept override { return size_; }

  /// @copydoc Abstract::operator==(const Abstract&) const
  bool operator==(const Abstract& rhs) const noexcept override {
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
