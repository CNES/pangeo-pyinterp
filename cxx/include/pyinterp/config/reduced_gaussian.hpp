// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "pyinterp/config/common.hpp"

namespace pyinterp::config::reduced_gaussian {

/// @brief When a column callback meets the time blend of a series.
enum class Blend : uint8_t {
  /// Blend the columns in time, then call the callback once.
  kColumns,
  /// Call the callback on the columns of each step, then blend the results.
  kResults,
};

/// @brief Configuration of the interpolation of reduced Gaussian grids.
class ReducedGaussian {
 public:
  /// @brief Default chunk size of the batched callbacks, in points.
  static constexpr int64_t kDefaultChunkSize = 65536;

  /// @brief Default constructor.
  constexpr ReducedGaussian() = default;

  /// @brief Common settings (bounds error, number of threads).
  [[nodiscard]] constexpr auto common() const noexcept -> const Common& {
    return common_;
  }

  /// @brief Interpolation in time between the steps of a series.
  [[nodiscard]] constexpr auto time_method() const noexcept -> AxisMethod {
    return time_method_;
  }

  /// @brief Width of the analysis intervals, or 0 for the steps that
  ///        bracket a date. Same unit as the dates of the series.
  [[nodiscard]] constexpr auto interval() const noexcept -> int64_t {
    return interval_;
  }

  /// @brief Largest time between the two steps a date reads, or 0 for no
  ///        limit. Same unit as the dates of the series.
  [[nodiscard]] constexpr auto max_step_gap() const noexcept -> int64_t {
    return max_step_gap_;
  }

  /// @brief When a callback meets the time blend.
  [[nodiscard]] constexpr auto blend() const noexcept -> Blend {
    return blend_;
  }

  /// @brief Number of points given to a batched callback at a time.
  [[nodiscard]] constexpr auto chunk_size() const noexcept -> int64_t {
    return chunk_size_;
  }

  /// @brief Update the `bounds_error` setting.
  [[nodiscard]] constexpr auto with_bounds_error(this ReducedGaussian self,
                                                 bool value)
      -> ReducedGaussian {
    self.common_ = self.common_.with_bounds_error(value);
    return self;
  }

  /// @brief Update the number of threads (0: all available).
  [[nodiscard]] constexpr auto with_num_threads(this ReducedGaussian self,
                                                size_t value)
      -> ReducedGaussian {
    self.common_ = self.common_.with_num_threads(value);
    return self;
  }

  /// @brief Update the interpolation in time.
  [[nodiscard]] constexpr auto with_time_method(this ReducedGaussian self,
                                                AxisMethod value)
      -> ReducedGaussian {
    self.time_method_ = value;
    return self;
  }

  /// @brief Update the width of the analysis intervals (0: none).
  [[nodiscard]] constexpr auto with_interval(this ReducedGaussian self,
                                             int64_t value) -> ReducedGaussian {
    if (value < 0) {
      throw std::invalid_argument("the interval must not be negative");
    }
    self.interval_ = value;
    return self;
  }

  /// @brief Update the largest time between two steps (0: no limit).
  [[nodiscard]] constexpr auto with_max_step_gap(this ReducedGaussian self,
                                                 int64_t value)
      -> ReducedGaussian {
    if (value < 0) {
      throw std::invalid_argument("max_step_gap must not be negative");
    }
    self.max_step_gap_ = value;
    return self;
  }

  /// @brief Update when a callback meets the time blend.
  [[nodiscard]] constexpr auto with_blend(this ReducedGaussian self,
                                          Blend value) -> ReducedGaussian {
    self.blend_ = value;
    return self;
  }

  /// @brief Update the number of points given to a batched callback.
  [[nodiscard]] constexpr auto with_chunk_size(this ReducedGaussian self,
                                               int64_t value)
      -> ReducedGaussian {
    if (value < 1) {
      throw std::invalid_argument("the chunk size must be positive");
    }
    self.chunk_size_ = value;
    return self;
  }

 private:
  Common common_{};
  AxisMethod time_method_{AxisMethod::kLinear};
  int64_t interval_{0};
  int64_t max_step_gap_{0};
  Blend blend_{Blend::kColumns};
  int64_t chunk_size_{kDefaultChunkSize};
};

}  // namespace pyinterp::config::reduced_gaussian
