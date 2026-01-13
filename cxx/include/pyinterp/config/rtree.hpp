// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cstdint>
#include <optional>

#include "pyinterp/config/common.hpp"
#include "pyinterp/geometry/rtree.hpp"
#include "pyinterp/math.hpp"
#include "pyinterp/math/interpolate/kriging.hpp"
#include "pyinterp/math/interpolate/rbf.hpp"
#include "pyinterp/math/interpolate/window_function.hpp"

namespace pyinterp::config::rtree {

/// @brief Base class for RTree interpolation configurations using CRTP.
/// @tparam Derived The derived configuration class.
/// @details Provides common functionality for all RTree-based interpolation
/// methods, including neighbor count (k), search radius, and thread settings.
template <typename Derived>
class RTreeBase : public ThreadConfig {
 public:
  /// @brief Default constructor
  constexpr RTreeBase() noexcept = default;

  /// @brief Get the number of neighbors.
  /// @return Number of neighbors to consider
  [[nodiscard]] constexpr auto k() const noexcept -> uint32_t { return k_; }

  /// @brief Get the search radius.
  /// @return Optional search radius in meters (nullopt = unlimited)
  [[nodiscard]] constexpr auto radius() const noexcept -> const double {
    return radius_;
  }

  /// @brief Get the type of boundary check to apply.
  /// @return Boundary check type
  [[nodiscard]] constexpr auto boundary_check() const noexcept
      -> geometry::BoundaryCheck {
    return boundary_check_;
  }

  /// @brief Set the number of neighbors
  /// @param[in] value Number of neighbors
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_k(this Derived self,
                                      uint32_t value) noexcept -> Derived {
    self.k_ = value;
    return self;
  }

  /// @brief Set the search radius in meters
  /// @param[in] value Search radius (nullopt for unlimited)
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_radius(
      this Derived self, const std::optional<double>& value) noexcept
      -> Derived {
    self.radius_ = value.value_or(std::numeric_limits<double>::max());
    return self;
  }

  /// @brief Set the number of threads
  /// @param[in] value Number of threads
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_num_threads(this Derived self,
                                                uint32_t value) noexcept
      -> Derived {
    auto base = static_cast<ThreadConfig&>(self).with_num_threads(value);
    static_cast<ThreadConfig&>(self) = base;
    return self;
  }

  /// @brief Set the boundary check type
  /// @param[in] value Boundary check type
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_boundary_check(
      this Derived self, geometry::BoundaryCheck value) noexcept -> Derived {
    self.boundary_check_ = value;
    return self;
  }

 protected:
  /// Type of boundary verification to apply
  geometry::BoundaryCheck boundary_check_;

  /// Number of neighbors to consider
  uint32_t k_{8};

  /// Optional search radius in meters (nullopt = unlimited)
  double radius_{std::numeric_limits<double>::max()};
};

// ////////////////////////////////////////////////////////////////////////////

/// Configuration for query operations
class Query : public RTreeBase<Query> {
 public:
  /// @brief Default constructor
  constexpr Query() noexcept = default;
};

// ////////////////////////////////////////////////////////////////////////////

/// Configuration for inverse distance weighting interpolation
class InverseDistanceWeighting : public RTreeBase<InverseDistanceWeighting> {
 public:
  /// @brief Default constructor
  constexpr InverseDistanceWeighting() noexcept = default;

  /// @brief Get the power parameter.
  /// @return Power parameter (exponent)
  [[nodiscard]] constexpr auto p() const noexcept -> uint32_t { return p_; }

  /// @brief Set the power parameter
  /// @param[in] value Power parameter
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_p(this InverseDistanceWeighting self,
                                      uint32_t value) noexcept
      -> InverseDistanceWeighting {
    self.p_ = value;
    return self;
  }

 private:
  /// Power parameter (exponent)
  uint32_t p_{2};
};

// ////////////////////////////////////////////////////////////////////////////

/// Configuration for kriging interpolation
class Kriging : public RTreeBase<Kriging> {
 public:
  /// @brief Default constructor
  constexpr Kriging() noexcept = default;

  /// @brief Get the sill parameter.
  /// @return Sill parameter (variance at infinity)
  [[nodiscard]] constexpr auto sigma() const noexcept -> double {
    return sigma_;
  }

  /// @brief Get the range parameter.
  /// @return Range parameter (distance scale)
  [[nodiscard]] constexpr auto lambda() const noexcept -> double {
    return lambda_;
  }
  /// @brief Get the nugget effect parameter.
  /// @return Nugget effect parameter (micro-scale variance)
  [[nodiscard]] constexpr auto nugget() const noexcept -> double {
    return nugget_;
  }

  /// @brief Get the covariance function type.
  /// @return Covariance function type
  [[nodiscard]] constexpr auto covariance_model() const noexcept
      -> math::interpolate::CovarianceFunction {
    return covariance_model_;
  }

  /// @brief Get the drift function.
  /// @return Optional drift function
  [[nodiscard]] constexpr auto drift_function() const noexcept
      -> const std::optional<math::interpolate::DriftFunction>& {
    return drift_function_;
  }

  /// @brief Set the sill parameter
  /// @param[in] value Sill parameter
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_sigma(this Kriging self,
                                          double value) noexcept -> Kriging {
    self.sigma_ = value;
    return self;
  }

  /// @brief Set the range parameter
  /// @param[in] value Range parameter
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_lambda(this Kriging self,
                                           double value) noexcept -> Kriging {
    self.lambda_ = value;
    return self;
  }

  /// @brief Set the nugget effect parameter
  /// @param[in] value Nugget effect parameter
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_nugget(this Kriging self,
                                           double value) noexcept -> Kriging {
    self.nugget_ = value;
    return self;
  }

  /// @brief Set the covariance model
  /// @param[in] value Covariance function type
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_covariance_model(
      this Kriging self, math::interpolate::CovarianceFunction value) noexcept
      -> Kriging {
    self.covariance_model_ = value;
    return self;
  }

  /// @brief Set the drift function
  /// @param[in] value Drift function type
  /// @return Updated configuration
  [[nodiscard]] auto with_drift_function(
      this Kriging self,
      std::optional<math::interpolate::DriftFunction> value) noexcept
      -> Kriging {
    self.drift_function_ = value;
    return self;
  }

 private:
  /// Sill parameter (variance at infinity)
  double sigma_{1.0};

  /// Range parameter (distance scale)
  double lambda_{1.0};

  /// Nugget effect parameter (micro-scale variance)
  double nugget_{0.0};

  /// Covariance function type
  math::interpolate::CovarianceFunction covariance_model_{
      math::interpolate::CovarianceFunction::kSpherical};

  /// Optional drift function
  std::optional<math::interpolate::DriftFunction> drift_function_;
};

// ////////////////////////////////////////////////////////////////////////////

/// Configuration for radial basis function interpolation
class RadialBasisFunction : public RTreeBase<RadialBasisFunction> {
 public:
  /// @brief Default constructor
  constexpr RadialBasisFunction() noexcept = default;

  /// @brief Get the radial basis function type.
  /// @return Radial basis function type
  [[nodiscard]] constexpr auto rbf() const noexcept
      -> math::interpolate::RBFKernel {
    return rbf_;
  }

  /// @brief Get the shape parameter (epsilon).
  /// @return Optional shape parameter (epsilon)
  [[nodiscard]] constexpr auto epsilon() const noexcept -> const double {
    return epsilon_;
  }

  /// @brief Get the smoothing parameter.
  /// @return Smoothing parameter
  [[nodiscard]] constexpr auto smooth() const noexcept -> double {
    return smooth_;
  }

  /// @brief Set the RBF type
  /// @param[in] value Radial basis function type
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_rbf(
      this RadialBasisFunction self,
      math::interpolate::RBFKernel value) noexcept -> RadialBasisFunction {
    self.rbf_ = value;
    return self;
  }

  /// @brief Set the shape parameter (epsilon)
  /// @param[in] value Shape parameter (nullopt for auto)
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_epsilon(
      this RadialBasisFunction self, std::optional<double> value) noexcept
      -> RadialBasisFunction {
    self.epsilon_ = value.value_or(std::numeric_limits<double>::quiet_NaN());
    return self;
  }

  /// @brief Set the smoothing parameter
  /// @param[in] value Smoothing parameter
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_smooth(this RadialBasisFunction self,
                                           double value) noexcept
      -> RadialBasisFunction {
    self.smooth_ = value;
    return self;
  }

 private:
  /// Radial basis function type
  math::interpolate::RBFKernel rbf_{
      math::interpolate::RBFKernel::kMultiquadric};

  /// Optional shape parameter (epsilon)
  double epsilon_{std::numeric_limits<double>::quiet_NaN()};

  /// Smoothing parameter
  double smooth_{0.0};
};

// ////////////////////////////////////////////////////////////////////////////

/// Configuration for window function interpolation
class InterpolationWindow : public RTreeBase<InterpolationWindow> {
 public:
  /// @brief Default constructor
  constexpr InterpolationWindow() noexcept = default;

  /// @brief Get the window function type.
  /// @return Window function type
  [[nodiscard]] constexpr auto wf() const noexcept
      -> math::interpolate::window::Kernel {
    return wf_;
  }

  /// @brief Get the window function argument.
  /// @return Optional window function argument
  [[nodiscard]] __CONSTEXPR auto arg() const noexcept -> double {
    if (!std::isnan(arg_)) {
      return arg_;
    }

    // The default argument value depends on the window function selected to
    // avoid invalid configurations.
    switch (wf_) {
      case math::interpolate::window::Kernel::kGaussian:
      case math::interpolate::window::Kernel::kLanczos:
        return 1.0;
      default:
        return 0.0;
    }
  }

  /// @brief Set the window function type
  /// @param[in] value Window function type
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_wf(
      this InterpolationWindow self,
      math::interpolate::window::Kernel value) noexcept -> InterpolationWindow {
    self.wf_ = value;
    return self;
  }

  /// @brief Set the window function argument
  /// @param[in] value Window function argument
  /// @return Updated configuration
  [[nodiscard]] constexpr auto with_arg(this InterpolationWindow self,
                                        std::optional<double> value) noexcept
      -> InterpolationWindow {
    self.arg_ = value.value_or(std::numeric_limits<double>::quiet_NaN());
    return self;
  }

 private:
  /// Window function type
  math::interpolate::window::Kernel wf_{
      math::interpolate::window::Kernel::kGaussian};

  /// Optional window function argument
  double arg_{std::numeric_limits<double>::quiet_NaN()};
};

}  // namespace pyinterp::config::rtree
