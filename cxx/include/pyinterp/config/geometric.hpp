// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <stdexcept>
#include <string_view>

#include "pyinterp/config/common.hpp"
#include "pyinterp/math/interpolate/geometric/bivariate.hpp"

namespace pyinterp::config {
namespace geometric {

/// Method for 2D spatial interpolation.
using SpatialMethod = math::interpolate::geometric::InterpolationMethod;

/// @brief Parser for a spatial interpolation method name.
/// @param[in] method_name Name of the interpolation method (case-sensitive)
/// @return Corresponding `SpatialMethod` enum value.
/// @throws std::invalid_argument if the method name is unknown.
[[nodiscard]] inline auto parse_spatial_method(std::string_view method_name)
    -> SpatialMethod {
  if (method_name == "bilinear") {
    return SpatialMethod::kBilinear;
  }
  if (method_name == "idw") {
    return SpatialMethod::kInverseDistanceWeighting;
  }
  if (method_name == "nearest") {
    return SpatialMethod::kNearest;
  }
  throw std::invalid_argument("Unknown spatial interpolation method: " +
                              std::string(method_name));
}

/// Configuration for 2D spatial interpolation.
class Spatial {
 public:
  /// @brief Default constructor
  constexpr Spatial() = default;

  /// @brief Constructor with an explicit interpolation method
  /// @param[in] method Interpolation method to use
  constexpr explicit Spatial(SpatialMethod method) : method_(method) {}

  /// @brief Constructor with method and exponent
  /// @param[in] method Interpolation method to use
  /// @param[in] exponent Exponent for the inverse distance weighting method
  constexpr Spatial(SpatialMethod method, int exponent)
      : method_(method), exponent_(exponent) {}

  /// @brief Get the interpolation method.
  /// @return Interpolation method
  [[nodiscard]] constexpr auto method() const -> SpatialMethod {
    return method_;
  }

  /// @brief Get the exponent for inverse distance weighting method.
  /// @return Exponent value
  [[nodiscard]] constexpr auto exponent() const -> int { return exponent_; }

  /// @brief Create a configuration for bilinear interpolation.
  /// @return `Spatial` configured with the bilinear interpolation method.
  [[nodiscard]] static constexpr auto bilinear() -> Spatial { return {}; }

  /// @brief Create a configuration for nearest-neighbor interpolation.
  /// @return `Spatial` configured with the nearest-neighbor method.
  [[nodiscard]] static constexpr auto nearest() -> Spatial {
    return Spatial{SpatialMethod::kNearest};
  }

  /// @brief Create a configuration for inverse distance weighting (IDW)
  /// interpolation.
  /// @param[in] exponent Exponent for the inverse distance weighting method
  /// (default: 2).
  /// @return `Spatial` configured with the IDW interpolation method.
  [[nodiscard]] static constexpr auto idw(int exponent = 2) -> Spatial {
    return Spatial{SpatialMethod::kInverseDistanceWeighting, exponent};
  }

 private:
  /// Interpolation method
  SpatialMethod method_{SpatialMethod::kBilinear};

  /// Exponent for inverse distance weighting method
  int exponent_{2};
};

// Forward declarations
class Bivariate;
class Trivariate;
class Quadrivariate;

}  // namespace geometric

/// @brief Traits to determine the number of additional axes for the
/// `Bivariate` configuration type.
template <>
struct InterpolationTraits<geometric::Bivariate> {
  static constexpr size_t num_axes = 0;
};

/// @brief Traits to determine the number of additional axes for the
/// `Trivariate` configuration type.
template <>
struct InterpolationTraits<geometric::Trivariate> {
  static constexpr size_t num_axes = 1;
};

/// @brief Traits to determine the number of additional axes for the
/// `Quadrivariate` configuration type.
template <>
struct InterpolationTraits<geometric::Quadrivariate> {
  static constexpr size_t num_axes = 2;
};

namespace geometric {

/// @brief Base class for bivariate-based interpolation configurations using
/// the Curiously Recurring Template Pattern (CRTP).
/// @tparam Derived The derived configuration class.
template <typename Derived>
struct BivariateBase : Base<Spatial, Derived> {
  /// @brief Create a configuration for bilinear interpolation
  [[nodiscard]] static constexpr auto bilinear() -> Derived {
    return create_config(Spatial::bilinear(), AxisConfig::linear());
  }

  /// @brief Create a configuration for IDW interpolation
  [[nodiscard]] static constexpr auto idw(int exp = 2) -> Derived {
    return create_config(Spatial::idw(exp), AxisConfig::linear());
  }

  /// @brief Create a configuration for nearest-neighbor interpolation
  [[nodiscard]] static constexpr auto nearest() -> Derived {
    return create_config(Spatial::nearest(), AxisConfig::nearest());
  }

 private:
  /// @brief Helper to create a configuration with the appropriate number
  /// of axes for the derived type.
  [[nodiscard]] static constexpr auto create_config(
      const Spatial& spatial, const AxisConfig& axis_config) -> Derived {
    constexpr size_t num_axes = InterpolationTraits<Derived>::num_axes;

    if constexpr (num_axes == 0) {
      // Bivariate case
      return Derived{spatial};
    } else if constexpr (num_axes == 1) {
      // Trivariate case
      return Derived{spatial, axis_config};
    } else if constexpr (num_axes == 2) {
      // Quadrivariate case
      return Derived{spatial, axis_config, axis_config};
    }
  }
};

/// Bivariate interpolation configuration (2D).
class Bivariate : public BivariateBase<Bivariate> {
 public:
  /// @brief Default constructor
  constexpr Bivariate() = default;

  /// @brief Constructor with spatial and common configurations
  /// @param[in] spatial Spatial interpolation configuration
  /// @param[in] common Common interpolation configuration (optional)
  constexpr explicit Bivariate(const Spatial& spatial,
                               const Common& common = {})
      : spatial_(spatial), common_(common) {}

  /// @brief Get the spatial interpolation configuration.
  /// @return Spatial interpolation configuration
  [[nodiscard]] constexpr auto spatial() const -> const Spatial& {
    return spatial_;
  }

  /// @brief Get the common interpolation configuration.
  /// @return Common interpolation configuration
  [[nodiscard]] constexpr auto common() const -> const Common& {
    return common_;
  }

  // Allow Base class to access members
  friend class Base<Spatial, Bivariate>;

 private:
  /// Spatial interpolation configuration
  Spatial spatial_;
  /// Common interpolation configuration
  Common common_;
};

/// Trivariate interpolation configuration (3D).
class Trivariate : public BivariateBase<Trivariate> {
 public:
  /// @brief Default constructor
  constexpr Trivariate() = default;

  /// @brief Constructor with spatial, third axis, and common configurations
  /// @param[in] spatial Spatial interpolation configuration
  /// @param[in] third_axis Third axis interpolation configuration
  /// @param[in] common Common interpolation configuration (optional)
  constexpr Trivariate(const Spatial& spatial, const AxisConfig& third_axis,
                       const Common& common = {})
      : spatial_(spatial), third_axis_(third_axis), common_(common) {}

  /// @brief Get the spatial interpolation configuration.
  /// @return Spatial interpolation configuration
  [[nodiscard]] constexpr auto spatial() const -> const Spatial& {
    return spatial_;
  }

  /// @brief Get the third axis interpolation configuration.
  /// @return Third axis interpolation configuration
  [[nodiscard]] constexpr auto third_axis() const -> const AxisConfig& {
    return third_axis_;
  }

  /// @brief Get the common interpolation configuration.
  /// @return Common interpolation configuration
  [[nodiscard]] constexpr auto common() const -> const Common& {
    return common_;
  }

  /// @brief Update the third-axis configuration.
  /// @param[in] config New third-axis configuration.
  /// @return Updated `Trivariate` instance.
  [[nodiscard]] constexpr auto with_third_axis(this Trivariate self,
                                               const AxisConfig& config)
      -> Trivariate {
    self.third_axis_ = config;
    return self;
  }

  // Allow Base class to access members
  friend class Base<Spatial, Trivariate>;

 private:
  /// Spatial interpolation configuration
  Spatial spatial_;
  /// Third axis interpolation configuration
  AxisConfig third_axis_;
  /// Common interpolation configuration
  Common common_;
};

/// Quadrivariate interpolation configuration (4D).
class Quadrivariate : public BivariateBase<Quadrivariate> {
 public:
  /// @brief Default constructor
  constexpr Quadrivariate() = default;

  /// @brief Constructor with spatial, third axis, fourth axis, and common
  /// configurations
  /// @param[in] spatial Spatial interpolation configuration
  /// @param[in] third_axis Third axis interpolation configuration
  /// @param[in] fourth_axis Fourth axis interpolation configuration
  /// @param[in] common Common interpolation configuration (optional)
  constexpr Quadrivariate(const Spatial& spatial, const AxisConfig& third_axis,
                          const AxisConfig& fourth_axis,
                          const Common& common = {})
      : spatial_(spatial),
        third_axis_(third_axis),
        fourth_axis_(fourth_axis),
        common_(common) {}

  /// @brief Get the spatial interpolation configuration.
  /// @return Spatial interpolation configuration
  [[nodiscard]] constexpr auto spatial() const -> const Spatial& {
    return spatial_;
  }

  /// @brief Get the third axis interpolation configuration.
  /// @return Third axis interpolation configuration
  [[nodiscard]] constexpr auto third_axis() const -> const AxisConfig& {
    return third_axis_;
  }

  /// @brief Get the fourth axis interpolation configuration.
  /// @return Fourth axis interpolation configuration
  [[nodiscard]] constexpr auto fourth_axis() const -> const AxisConfig& {
    return fourth_axis_;
  }

  /// @brief Get the common interpolation configuration.
  /// @return Common interpolation configuration
  [[nodiscard]] constexpr auto common() const -> const Common& {
    return common_;
  }

  /// @brief Update the third-axis configuration.
  /// @param[in] config New third-axis configuration.
  /// @return Updated `Quadrivariate` instance.
  [[nodiscard]] constexpr auto with_third_axis(this Quadrivariate self,
                                               const AxisConfig& config)
      -> Quadrivariate {
    self.third_axis_ = config;
    return self;
  }

  /// @brief Update the fourth-axis configuration.
  /// @param[in] config New fourth-axis configuration.
  /// @return Updated `Quadrivariate` instance.
  [[nodiscard]] constexpr auto with_fourth_axis(this Quadrivariate self,
                                                const AxisConfig& config)
      -> Quadrivariate {
    self.fourth_axis_ = config;
    return self;
  }

  // Allow Base class to access members
  friend class Base<Spatial, Quadrivariate>;

 private:
  /// Spatial interpolation configuration
  Spatial spatial_;
  /// Third axis interpolation configuration
  AxisConfig third_axis_;
  /// Fourth axis interpolation configuration
  AxisConfig fourth_axis_;
  /// Common interpolation configuration
  Common common_;
};

}  // namespace geometric
}  // namespace pyinterp::config
