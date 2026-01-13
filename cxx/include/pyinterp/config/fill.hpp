// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include "pyinterp/config/common.hpp"
#include "pyinterp/math.hpp"

namespace pyinterp::config::fill {

/// Initial guess strategy for iterative fill methods
enum class FirstGuess : uint8_t {
  /// Use zonal average of defined values
  kZonalAverage,
  /// Use zero as initial guess
  kZero,
};

/// Type of values processed by the LOESS filter.
enum class LoessValueType : std::uint8_t {
  kUndefined,  ///< Fill only undefined (NaN) values
  kDefined,    ///< Smooth only defined values
  kAll         ///< Smooth and fill all values
};

/// @brief Parser for a first guess strategy name.
/// @param[in] strategy_name Name of the strategy (case-sensitive)
/// @return Corresponding FirstGuess enum value
/// @throws std::invalid_argument if the strategy name is unknown
[[nodiscard]] inline auto parse_first_guess(std::string_view strategy_name)
    -> FirstGuess {
  if (strategy_name == "zonal_average") {
    return FirstGuess::kZonalAverage;
  }
  if (strategy_name == "zero") {
    return FirstGuess::kZero;
  }
  throw std::invalid_argument("Unknown first guess strategy: " +
                              std::string(strategy_name));
}

/// @brief Parser for a loess value type name.
/// @param[in] value_type_name Name of the value type (case-sensitive)
/// @return Corresponding LoessValueType enum value
/// @throws std::invalid_argument if the value type name is unknown
[[nodiscard]] inline auto parse_loess_value_type(
    std::string_view value_type_name) -> LoessValueType {
  if (value_type_name == "undefined") {
    return LoessValueType::kUndefined;
  }
  if (value_type_name == "defined") {
    return LoessValueType::kDefined;
  }
  if (value_type_name == "all") {
    return LoessValueType::kAll;
  }
  throw std::invalid_argument("Unknown LOESS value type: " +
                              std::string(value_type_name));
}

/// @brief Base class for fill method configurations using CRTP.
/// @tparam Derived The derived configuration class.
/// @details Provides common functionality for all fill methods,
/// including first guess strategy, circularity, iterations, and convergence.
template <typename Derived>
class FillBase : public ThreadConfig {
 public:
  /// @brief Default constructor
  constexpr FillBase() noexcept = default;

  /// @brief Get the first guess strategy.
  /// @return First guess strategy
  [[nodiscard]] constexpr auto first_guess() const noexcept -> FirstGuess {
    return first_guess_;
  }

  /// @brief Get whether the grid is periodic (e.g., longitude).
  /// @return True if grid is periodic
  [[nodiscard]] constexpr auto is_periodic() const noexcept -> bool {
    return is_periodic_;
  }

  /// @brief Get the maximum number of iterations.
  /// @return Maximum iterations
  [[nodiscard]] constexpr auto max_iterations() const noexcept -> uint32_t {
    return max_iterations_;
  }

  /// @brief Get the convergence threshold (epsilon).
  /// @return Epsilon value
  [[nodiscard]] constexpr auto epsilon() const noexcept -> double {
    return epsilon_;
  }

  /// @brief Set the first guess strategy
  /// @param[in] value First guess strategy
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_first_guess(this Derived self,
                                                FirstGuess value) noexcept
      -> Derived {
    self.first_guess_ = value;
    return self;
  }

  /// @brief Set the is_periodic setting
  /// @param[in] value Whether grid is periodic
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_is_periodic(this Derived self,
                                                bool value) noexcept
      -> Derived {
    self.is_periodic_ = value;
    return self;
  }

  /// @brief Set the maximum iterations
  /// @param[in] value Maximum iterations
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_max_iterations(this Derived self,
                                                   uint32_t value) noexcept
      -> Derived {
    self.max_iterations_ = value;
    return self;
  }

  /// @brief Set the convergence threshold
  /// @param[in] value Epsilon value
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_epsilon(

      this Derived self, double value) noexcept -> Derived {
    self.epsilon_ = value;
    return self;
  }

  /// @brief Set the number of threads
  /// @param[in] value Number of threads
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_num_threads(this Derived self,
                                                size_t value) noexcept
      -> Derived {
    auto base = static_cast<ThreadConfig&>(self).with_num_threads(value);
    static_cast<ThreadConfig&>(self) = base;
    return self;
  }

 protected:
  /// First guess strategy
  FirstGuess first_guess_{FirstGuess::kZonalAverage};
  /// X-Axis is it circular (e.g., longitude)
  bool is_periodic_{true};
  /// Maximum iterations
  uint32_t max_iterations_{500};
  /// Convergence threshold
  double epsilon_{1e-4};
};

/// Configuration for LOESS fill method
class Loess : public FillBase<Loess> {
 public:
  /// @brief Default constructor
  constexpr Loess() noexcept = default;

  /// @brief Get the value type to process.
  /// @return Value type
  [[nodiscard]] constexpr auto value_type() const noexcept -> LoessValueType {
    return value_type_;
  }

  /// @brief Set the value type to process
  /// @param[in] value Value type
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_value_type(this Loess self,
                                               LoessValueType value) noexcept
      -> Loess {
    self.value_type_ = value;
    return self;
  }

  /// @brief Get the half-window size along x-axis.
  /// @return Half-window size along x-axis
  [[nodiscard]] constexpr auto nx() const noexcept -> uint32_t { return nx_; }

  /// @brief Set the half-window size along x-axis
  /// @param[in] value Half-window size along x-axis
  /// @return Updated configuration
  /// @throws std::invalid_argument if value is zero
  [[nodiscard]] __CONSTEXPR auto with_nx(this Loess self, uint32_t value)
      -> Loess {
    Loess::check_windows_size(value);
    self.nx_ = value;
    return self;
  }

  /// @brief Get the half-window size along y-axis.
  /// @return Half-window size along y-axis
  [[nodiscard]] constexpr auto ny() const noexcept -> uint32_t { return ny_; }

  /// @brief Set the half-window size along y-axis
  /// @param[in] value Half-window size along y-axis
  /// @return Updated configuration
  /// @throws std::invalid_argument if value is zero
  [[nodiscard]] __CONSTEXPR auto with_ny(this Loess self, uint32_t value)
      -> Loess {
    Loess::check_windows_size(value);
    self.ny_ = value;
    return self;
  }

 private:
  LoessValueType value_type_{LoessValueType::kUndefined};
  uint32_t nx_{3};
  uint32_t ny_{3};

  /// Check that window sizes are valid
  static void check_windows_size(uint32_t value) {
    if (value == 0) {
      throw std::invalid_argument("Window size must be greater than zero.");
    }
  }
};

/// Configuration for FFT Inpaint fill method
class FFTInpaint : public FillBase<FFTInpaint> {
 public:
  /// @brief Default constructor
  constexpr FFTInpaint() noexcept { max_iterations_ = 500; }

  /// @brief Get the sigma parameter.
  /// @return Sigma value
  [[nodiscard]] constexpr auto sigma() const noexcept -> double {
    return sigma_;
  }

  /// @brief Set the sigma parameter
  /// @param[in] value Sigma value
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_sigma(this FFTInpaint self,
                                          double value) noexcept -> FFTInpaint {
    self.sigma_ = value;
    return self;
  }

 private:
  /// Sigma parameter for FFT
  double sigma_{10.0};
};

/// Configuration for Gauss-Seidel fill method
class GaussSeidel : public FillBase<GaussSeidel> {
 public:
  /// @brief Default constructor
  constexpr GaussSeidel() noexcept { max_iterations_ = 2000; }

  /// @brief Get the relaxation parameter.
  /// @return Relaxation value
  [[nodiscard]] constexpr auto relaxation() const noexcept -> double {
    return relaxation_;
  }

  /// @brief Set the relaxation parameter
  /// @param[in] value Relaxation value
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_relaxation(this GaussSeidel self,
                                               double value) noexcept
      -> GaussSeidel {
    self.relaxation_ = value;
    return self;
  }

 private:
  /// Relaxation parameter (0 < relaxation <= 2, typically 1.0 for Gauss-Seidel)
  double relaxation_{1.0};
};

/// Configuration for Multigrid fill method
class Multigrid : public FillBase<Multigrid> {
 public:
  /// @brief Default constructor
  constexpr Multigrid() noexcept { max_iterations_ = 100; }

  /// @brief Get the number of pre-smoothing iterations.
  /// @return Pre-smoothing iterations
  [[nodiscard]] constexpr auto pre_smooth() const noexcept -> uint32_t {
    return pre_smooth_;
  }

  /// @brief Get the number of post-smoothing iterations.
  /// @return Post-smoothing iterations
  [[nodiscard]] constexpr auto post_smooth() const noexcept -> uint32_t {
    return post_smooth_;
  }

  /// @brief Set the number of pre-smoothing iterations
  /// @param[in] value Pre-smoothing iterations
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_pre_smooth(this Multigrid self,
                                               uint32_t value) noexcept
      -> Multigrid {
    self.pre_smooth_ = value;
    return self;
  }

  /// @brief Set the number of post-smoothing iterations
  /// @param[in] value Post-smoothing iterations
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_post_smooth(this Multigrid self,
                                                uint32_t value) noexcept
      -> Multigrid {
    self.post_smooth_ = value;
    return self;
  }

 private:
  /// Number of pre-smoothing iterations
  uint32_t pre_smooth_{2};
  /// Number of post-smoothing iterations
  uint32_t post_smooth_{2};
};

}  // namespace pyinterp::config::fill
