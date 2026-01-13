// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <stdexcept>
#include <string_view>
#include <variant>

#include "pyinterp/config/common.hpp"
#include "pyinterp/math/axis.hpp"
#include "pyinterp/math/interpolate/factory.hpp"

namespace pyinterp::config {
namespace windowed {

/// Type alias for boundary mode
using BoundaryMode = math::axis::Boundary;

/// Type alias for Spline fitting model
using Spline = math::interpolate::univariate::Method;

/// Type alias for Bicubic fitting model
using Bicubic = math::interpolate::bivariate::Method;

/// @brief Known fitting models
using SpatialMethod = std::variant<Spline, Bicubic>;

/// @brief Parser for a fitting model name.
/// @param[in] model_name Name of the fitting model (case-sensitive)
/// @return Corresponding `SpatialMethod` variant value.
/// @throws std::invalid_argument if the model name is unknown.
[[nodiscard]] inline auto parse_fitting_model(std::string_view model_name)
    -> SpatialMethod {
  if (model_name == "akima") {
    return Spline::kAkima;
  }
  if (model_name == "akima_periodic") {
    return Spline::kAkimaPeriodic;
  }
  if (model_name == "c_spline") {
    return Spline::kCSpline;
  }
  if (model_name == "c_spline_not_a_knot") {
    return Spline::kCSplineNotAKnot;
  }
  if (model_name == "c_spline_periodic") {
    return Spline::kCSplinePeriodic;
  }
  if (model_name == "linear") {
    return Spline::kLinear;
  }
  if (model_name == "steffen") {
    return Spline::kSteffen;
  }
  if (model_name == "polynomial") {
    return Spline::kPolynomial;
  }
  if (model_name == "bilinear") {
    return Bicubic::kBilinear;
  }
  if (model_name == "bicubic") {
    return Bicubic::kBicubic;
  }
  throw std::invalid_argument("Unknown fitting model: " +
                              std::string(model_name));
}

/// @brief Configuration for boundary handling
///
/// Because only a limited set of boundary modes is supported, we expose
/// `BoundaryConfig` helpers rather than the raw `math::axis::Boundary` enum
/// to prevent invalid inputs.
class BoundaryConfig {
 public:
  /// @brief Default constructor
  constexpr BoundaryConfig() = default;

  /// @brief Constructor with an explicit boundary mode
  /// @param[in] mode Boundary mode to use
  constexpr explicit BoundaryConfig(BoundaryMode mode) : mode_(mode) {}

  /// @brief Get the boundary mode.
  /// @return Boundary mode
  [[nodiscard]] constexpr auto mode() const -> BoundaryMode { return mode_; }

  /// @brief Create a configuration for undefined boundary mode.
  /// @return `BoundaryConfig` configured with the undefined boundary mode
  [[nodiscard]] static constexpr auto undef() -> BoundaryConfig {
    return BoundaryConfig{BoundaryMode::kUndef};
  }

  /// @brief Create a configuration for shrink boundary mode.
  /// @return `BoundaryConfig` configured with the shrink boundary mode
  [[nodiscard]] static constexpr auto shrink() -> BoundaryConfig {
    return BoundaryConfig{BoundaryMode::kShrink};
  }

 private:
  /// Boundary mode
  BoundaryMode mode_{BoundaryMode::kUndef};
};

/// @brief Configuration for 2D spatial interpolation.
class Spatial {
 public:
  /// @brief Default constructor
  constexpr Spatial() = default;

  /// @brief Constructor with an explicit interpolation method
  /// @param[in] method Interpolation method to use
  constexpr explicit Spatial(SpatialMethod method) : method_(method) {}

  /// @brief Get the interpolation method.
  /// @return Interpolation method
  [[nodiscard]] constexpr auto method() const -> const SpatialMethod& {
    return method_;
  }

  /// @brief Get the boundary mode.
  /// @return Boundary mode
  [[nodiscard]] constexpr auto boundary_mode() const -> BoundaryMode {
    return boundary_mode_;
  }

  /// @brief Get the half window size in x direction.
  /// @return Half window size in x direction
  [[nodiscard]] constexpr auto half_window_size_x() const -> int {
    return half_window_size_x_;
  }

  /// @brief Get the half window size in y direction.
  /// @return Half window size in y direction
  [[nodiscard]] constexpr auto half_window_size_y() const -> int {
    return half_window_size_y_;
  }

  /// @brief Create the interpolator instance.
  /// @tparam T Data type handled by the interpolator.
  /// @return Unique pointer to the interpolator instance.
  template <typename T>
  [[nodiscard]] auto factory() const
      -> std::unique_ptr<math::interpolate::BivariateBase<T>> {
    return std::visit([](auto&& m) { return math::interpolate::factory<T>(m); },
                      method_);
  }

  /// @brief Create a configuration for bilinear interpolation.
  /// @return `Spatial` configured with the bilinear interpolation method.
  [[nodiscard]] static constexpr auto bilinear() -> Spatial {
    return Spatial{Bicubic::kBilinear};
  }

  /// @brief Create a configuration for bicubic interpolation.
  /// @return `Spatial` configured with the bicubic interpolation method.
  [[nodiscard]] static constexpr auto bicubic() -> Spatial {
    return Spatial{Bicubic::kBicubic};
  }

  /// @brief Create a configuration for linear interpolation.
  /// @return `Spatial` configured with the linear interpolation method.
  [[nodiscard]] static constexpr auto linear() -> Spatial {
    return Spatial{Spline::kLinear};
  }

  /// @brief Create a configuration for Akima spline interpolation.
  /// @return `Spatial` configured with the Akima spline interpolation method.
  [[nodiscard]] static constexpr auto akima() -> Spatial {
    return Spatial{Spline::kAkima};
  }

  /// @brief Create a configuration for Akima periodic spline interpolation.
  /// @return `Spatial` configured with the Akima periodic spline method.
  [[nodiscard]] static constexpr auto akima_periodic() -> Spatial {
    return Spatial{Spline::kAkimaPeriodic};
  }

  /// @brief Create a configuration for a C-spline interpolation.
  /// @return `Spatial` configured with the C-spline interpolation method.
  [[nodiscard]] static constexpr auto c_spline() -> Spatial {
    return Spatial{Spline::kCSpline};
  }

  /// @brief Create a configuration for C-spline (not-a-knot) interpolation.
  /// @return `Spatial` configured with the not-a-knot C-spline method.
  [[nodiscard]] static constexpr auto c_spline_not_a_knot() -> Spatial {
    return Spatial{Spline::kCSplineNotAKnot};
  }

  /// @brief Create a configuration for C-spline periodic interpolation.
  [[nodiscard]] static constexpr auto c_spline_periodic() -> Spatial {
    return Spatial{Spline::kCSplinePeriodic};
  }

  /// @brief Create a configuration for Steffen spline interpolation.
  /// @return `Spatial` configured with the Steffen spline method.
  [[nodiscard]] static constexpr auto steffen() -> Spatial {
    return Spatial{Spline::kSteffen};
  }

  /// @brief Create a configuration for polynomial spline interpolation.
  /// @return `Spatial` configured with the polynomial spline method.
  [[nodiscard]] static constexpr auto polynomial() -> Spatial {
    return Spatial{Spline::kPolynomial};
  }

  /// @brief Update the `boundary_mode` setting.
  /// @param[in] config New boundary mode.
  /// @return Updated `Spatial` instance with the new setting.
  [[nodiscard]] constexpr auto with_boundary_mode(this Spatial self,
                                                  BoundaryConfig config)
      -> Spatial {
    self.boundary_mode_ = config.mode();
    return self;
  }

  /// @brief Update half window size in the x direction.
  /// @param[in] size New half window size in x direction.
  /// @return Updated `Spatial` instance with the new half window size.
  [[nodiscard]] constexpr auto with_half_window_size_x(this Spatial self,
                                                       int size) -> Spatial {
    self.half_window_size_x_ = size;
    return self;
  }

  /// @brief Update half window size in the y direction.
  /// @param[in] size New half window size in y direction.
  /// @return Updated `Spatial` instance with the new half window size.
  [[nodiscard]] constexpr auto with_half_window_size_y(this Spatial self,
                                                       int size) -> Spatial {
    self.half_window_size_y_ = size;
    return self;
  }

 private:
  /// Interpolation method
  SpatialMethod method_{Bicubic::kBicubic};

  /// Boundary mode
  BoundaryMode boundary_mode_{BoundaryMode::kUndef};

  /// Window size in x direction
  int half_window_size_x_{3};

  /// Window size in y direction
  int half_window_size_y_{3};
};

// Forward declarations
class Univariate;
class Bivariate;
class Trivariate;
class Quadrivariate;

}  // namespace windowed

/// @brief Traits to determine the number of additional axes for the
/// `Univariate` configuration type.
template <>
struct InterpolationTraits<windowed::Univariate> {
  static constexpr size_t num_axes = 0;
};

/// @brief Traits to determine the number of additional axes for the
/// `Bivariate` (windowed) configuration type.
template <>
struct InterpolationTraits<windowed::Bivariate> {
  static constexpr size_t num_axes = 0;
};

/// @brief Traits to determine the number of additional axes for the
/// `Trivariate` (windowed) configuration type.
template <>
struct InterpolationTraits<windowed::Trivariate> {
  static constexpr size_t num_axes = 1;
};

/// @brief Traits to determine the number of additional axes for the
/// `Quadrivariate` (windowed) configuration type.
template <>
struct InterpolationTraits<windowed::Quadrivariate> {
  static constexpr size_t num_axes = 2;
};

namespace windowed {

/// @brief Configuration for 1D univariate interpolation method
class UnivariateMethod {
 public:
  /// @brief Default constructor
  constexpr UnivariateMethod() = default;

  /// @brief Constructor with an explicit interpolation method
  /// @param[in] method Interpolation method to use
  constexpr explicit UnivariateMethod(Spline method) : method_(method) {}

  /// @brief Get the interpolation method.
  /// @return Interpolation method
  [[nodiscard]] constexpr auto method() const -> Spline { return method_; }

  /// @brief Get the boundary mode.
  /// @return Boundary mode
  [[nodiscard]] constexpr auto boundary_mode() const -> BoundaryMode {
    return boundary_mode_;
  }

  /// @brief Get the half window size.
  /// @return Half window size
  [[nodiscard]] constexpr auto half_window_size() const -> int {
    return half_window_size_;
  }

  /// @brief Create the interpolator instance.
  /// @tparam T Data type handled by the interpolator.
  /// @return Unique pointer to the interpolator instance.
  template <typename T>
  [[nodiscard]] auto factory() const
      -> std::unique_ptr<math::interpolate::Univariate<T>> {
    return math::interpolate::univariate::factory<T>(method_);
  }

  /// @brief Create a configuration for linear interpolation.
  [[nodiscard]] static constexpr auto linear() -> UnivariateMethod {
    return UnivariateMethod{Spline::kLinear};
  }

  /// @brief Create a configuration for Akima spline interpolation.
  [[nodiscard]] static constexpr auto akima() -> UnivariateMethod {
    return UnivariateMethod{Spline::kAkima};
  }

  /// @brief Create a configuration for Akima periodic spline interpolation.
  [[nodiscard]] static constexpr auto akima_periodic() -> UnivariateMethod {
    return UnivariateMethod{Spline::kAkimaPeriodic};
  }

  /// @brief Create a configuration for cubic spline interpolation.
  [[nodiscard]] static constexpr auto c_spline() -> UnivariateMethod {
    return UnivariateMethod{Spline::kCSpline};
  }

  /// @brief Create a configuration for cubic spline not-a-knot interpolation.
  [[nodiscard]] static constexpr auto c_spline_not_a_knot()
      -> UnivariateMethod {
    return UnivariateMethod{Spline::kCSplineNotAKnot};
  }

  /// @brief Create a configuration for cubic spline periodic interpolation.
  [[nodiscard]] static constexpr auto c_spline_periodic() -> UnivariateMethod {
    return UnivariateMethod{Spline::kCSplinePeriodic};
  }

  /// @brief Create a configuration for Steffen spline interpolation.
  [[nodiscard]] static constexpr auto steffen() -> UnivariateMethod {
    return UnivariateMethod{Spline::kSteffen};
  }

  /// @brief Create a configuration for polynomial interpolation.
  [[nodiscard]] static constexpr auto polynomial() -> UnivariateMethod {
    return UnivariateMethod{Spline::kPolynomial};
  }

  /// @brief Update the `boundary_mode` setting.
  /// @param[in] config New boundary mode.
  /// @return Updated instance with the new setting.
  [[nodiscard]] constexpr auto with_boundary_mode(this UnivariateMethod self,
                                                  BoundaryConfig config)
      -> UnivariateMethod {
    self.boundary_mode_ = config.mode();
    return self;
  }

  /// @brief Update half window size.
  /// @param[in] size New half window size.
  /// @return Updated instance with the new half window size.
  [[nodiscard]] constexpr auto with_half_window_size(this UnivariateMethod self,
                                                     int size)
      -> UnivariateMethod {
    self.half_window_size_ = size;
    return self;
  }

 private:
  /// Interpolation method
  Spline method_{Spline::kLinear};

  /// Boundary mode
  BoundaryMode boundary_mode_{BoundaryMode::kUndef};

  /// Window size
  int half_window_size_{3};
};

/// @brief Mixin providing common configuration modifiers.
template <typename Derived>
class CommonModifiers {
 public:
  /// @brief Update the `bounds_error` setting on the derived config.
  /// @param[in] value New value for `bounds_error`.
  /// @return Updated derived configuration instance.
  [[nodiscard]] constexpr auto with_bounds_error(this Derived self, bool value)
      -> Derived {
    self.common_ = self.common_.with_bounds_error(value);
    return self;
  }

  /// @brief Update the `num_threads` setting on the derived config.
  /// @param[in] value New value for `num_threads`.
  /// @return Updated derived configuration instance.
  [[nodiscard]] constexpr auto with_num_threads(this Derived self, size_t value)
      -> Derived {
    self.common_ = self.common_.with_num_threads(value);
    return self;
  }
};

/// @brief Mixin providing spatial window modifiers.
template <typename Derived>
class SpatialModifiers {
 public:
  /// @brief Update the `half_window_size_x` setting on the derived config.
  /// @param[in] size New half window size for the x-axis.
  /// @return Updated derived configuration instance.
  [[nodiscard]] constexpr auto with_half_window_size_x(this Derived self,
                                                       int size) -> Derived {
    self.spatial_ = self.spatial_.with_half_window_size_x(size);
    return self;
  }

  /// @brief Update the `half_window_size_y` setting on the derived config.
  /// @param[in] size New half window size for the y-axis.
  /// @return Updated derived configuration instance.
  [[nodiscard]] constexpr auto with_half_window_size_y(this Derived self,
                                                       int size) -> Derived {
    self.spatial_ = self.spatial_.with_half_window_size_y(size);
    return self;
  }

  /// @brief Update the `boundary_mode` setting on the derived config.
  /// @param[in] config New boundary mode.
  /// @return Updated derived configuration instance.
  [[nodiscard]] constexpr auto with_boundary_mode(this Derived self,
                                                  BoundaryConfig config)
      -> Derived {
    self.spatial_ = self.spatial_.with_boundary_mode(config);
    return self;
  }
};

/// @brief Mixin providing static factory methods for windowed interpolation.
/// Static methods ARE inherited, so derived classes get them automatically!
template <typename Derived>
class WindowedFactories {
 private:
  /// @brief Helper to create a Derived instance with default axis configs.
  /// @param[in] spatial Spatial interpolation configuration
  /// @return Derived instance
  [[nodiscard]] static constexpr auto make(const Spatial& spatial) -> Derived {
    constexpr size_t num_axes = InterpolationTraits<Derived>::num_axes;
    auto axis = AxisConfig::linear();

    if constexpr (num_axes == 0) {
      return Derived{spatial};
    } else if constexpr (num_axes == 1) {
      return Derived{spatial, axis};
    } else if constexpr (num_axes == 2) {
      return Derived{spatial, axis, axis};
    }
  }

 public:
  /// @brief Create a configuration for Akima spline interpolation.
  /// @return `Derived` configured with the Akima spline interpolation method.
  [[nodiscard]] static constexpr auto akima() -> Derived {
    return make(Spatial::akima());
  }

  /// @brief Create a configuration for Akima periodic spline interpolation.
  /// @return `Derived` configured with the Akima periodic spline method.
  [[nodiscard]] static constexpr auto akima_periodic() -> Derived {
    return make(Spatial::akima_periodic());
  }

  /// @brief Create a configuration for a C-spline interpolation.
  /// @return `Derived` configured with the C-spline interpolation method.
  [[nodiscard]] static constexpr auto c_spline() -> Derived {
    return make(Spatial::c_spline());
  }

  /// @brief Create a configuration for C-spline (not-a-knot) interpolation.
  /// @return `Derived` configured with the not-a-knot C-spline method.
  [[nodiscard]] static constexpr auto c_spline_not_a_knot() -> Derived {
    return make(Spatial::c_spline_not_a_knot());
  }

  /// @brief Create a configuration for C-spline periodic interpolation.
  /// @return `Derived` configured with the C-spline periodic method.
  [[nodiscard]] static constexpr auto c_spline_periodic() -> Derived {
    return make(Spatial::c_spline_periodic());
  }

  /// @brief Create a configuration for Steffen spline interpolation.
  /// @return `Derived` configured with the Steffen spline method.
  [[nodiscard]] static constexpr auto steffen() -> Derived {
    return make(Spatial::steffen());
  }

  /// @brief Create a configuration for linear interpolation.
  /// @return `Derived` configured with the linear interpolation method.
  [[nodiscard]] static constexpr auto linear() -> Derived {
    return make(Spatial::linear());
  }

  /// @brief Create a configuration for polynomial interpolation.
  /// @return `Derived` configured with the polynomial interpolation method.
  [[nodiscard]] static constexpr auto polynomial() -> Derived {
    return make(Spatial::polynomial());
  }

  /// @brief Create a configuration for bilinear interpolation.
  /// @return `Derived` configured with the bilinear interpolation method.
  [[nodiscard]] static constexpr auto bilinear() -> Derived {
    return make(Spatial::bilinear());
  }

  /// @brief Create a configuration for bicubic interpolation.
  /// @return `Derived` configured with the bicubic interpolation method.
  [[nodiscard]] static constexpr auto bicubic() -> Derived {
    return make(Spatial::bicubic());
  }
};

/// @brief Univariate windowed interpolation configuration (1D)
class Univariate : public CommonModifiers<Univariate> {
 public:
  /// @brief Default constructor
  constexpr Univariate() = default;

  /// @brief Constructor with univariate method and common configurations
  /// @param[in] univariate Univariate method configuration
  /// @param[in] common Common interpolation configuration
  constexpr explicit Univariate(const UnivariateMethod& univariate,
                                const Common& common = {})
      : univariate_(univariate), common_(common) {}

  /// @brief Get the univariate method configuration.
  /// @return Univariate method configuration
  [[nodiscard]] constexpr auto univariate() const -> const UnivariateMethod& {
    return univariate_;
  }

  /// @brief Get the common interpolation configuration.
  /// @return Common interpolation configuration
  [[nodiscard]] constexpr auto common() const -> const Common& {
    return common_;
  }

  /// @brief Create a configuration for linear interpolation.
  [[nodiscard]] static constexpr auto linear() -> Univariate {
    return Univariate{UnivariateMethod::linear()};
  }

  /// @brief Create a configuration for Akima spline interpolation.
  [[nodiscard]] static constexpr auto akima() -> Univariate {
    return Univariate{UnivariateMethod::akima()};
  }

  /// @brief Create a configuration for Akima periodic spline interpolation.
  [[nodiscard]] static constexpr auto akima_periodic() -> Univariate {
    return Univariate{UnivariateMethod::akima_periodic()};
  }

  /// @brief Create a configuration for cubic spline interpolation.
  [[nodiscard]] static constexpr auto c_spline() -> Univariate {
    return Univariate{UnivariateMethod::c_spline()};
  }

  /// @brief Create a configuration for cubic spline not-a-knot interpolation.
  [[nodiscard]] static constexpr auto c_spline_not_a_knot() -> Univariate {
    return Univariate{UnivariateMethod::c_spline_not_a_knot()};
  }

  /// @brief Create a configuration for cubic spline periodic interpolation.
  [[nodiscard]] static constexpr auto c_spline_periodic() -> Univariate {
    return Univariate{UnivariateMethod::c_spline_periodic()};
  }

  /// @brief Create a configuration for Steffen spline interpolation.
  [[nodiscard]] static constexpr auto steffen() -> Univariate {
    return Univariate{UnivariateMethod::steffen()};
  }

  /// @brief Create a configuration for polynomial interpolation.
  [[nodiscard]] static constexpr auto polynomial() -> Univariate {
    return Univariate{UnivariateMethod::polynomial()};
  }

  /// @brief Update window size.
  /// @param[in] size New window size.
  /// @return Updated `Univariate` instance with the new window size.
  [[nodiscard]] constexpr auto with_half_window_size(this Univariate self,
                                                     int size) -> Univariate {
    self.univariate_ = self.univariate_.with_half_window_size(size);
    return self;
  }

  /// @brief Update the `boundary_mode` setting.
  /// @param[in] config New boundary mode.
  /// @return Updated `Univariate` instance with the new setting.
  [[nodiscard]] constexpr auto with_boundary_mode(this Univariate self,
                                                  BoundaryConfig config)
      -> Univariate {
    self.univariate_ = self.univariate_.with_boundary_mode(config);
    return self;
  }

 private:
  friend class CommonModifiers<Univariate>;

  /// Univariate method configuration
  UnivariateMethod univariate_;
  /// Common interpolation configuration
  Common common_;
};

/// @brief Window interpolation configuration (2D only)
class Bivariate : public CommonModifiers<Bivariate>,
                  public SpatialModifiers<Bivariate>,
                  public WindowedFactories<Bivariate> {
 public:
  /// @brief Default constructor
  constexpr Bivariate() = default;

  /// @brief Constructor with spatial and common configurations
  /// @param[in] spatial Spatial interpolation configuration
  /// @param[in] common Common interpolation configuration
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

 private:
  friend class CommonModifiers<Bivariate>;
  friend class SpatialModifiers<Bivariate>;
  friend class WindowedFactories<Bivariate>;

  /// Spatial interpolation configuration
  Spatial spatial_;
  /// Common interpolation configuration
  Common common_;
};

/// @brief Windows interpolation configuration (3D only)
class Trivariate : public CommonModifiers<Trivariate>,
                   public SpatialModifiers<Trivariate>,
                   public WindowedFactories<Trivariate> {
 public:
  /// @brief Default constructor
  constexpr Trivariate() = default;

  /// @brief Constructor with spatial, third axis, and common configurations
  /// @param[in] spatial Spatial interpolation configuration
  /// @param[in] third_axis Third axis interpolation configuration
  /// @param[in] common Common interpolation configuration
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

  /// @brief Update third axis configuration
  /// @param[in] config New third axis configuration
  /// @return Updated TrivariateConfig instance
  [[nodiscard]] constexpr auto with_third_axis(this Trivariate self,
                                               const AxisConfig& config)
      -> Trivariate {
    self.third_axis_ = config;
    return self;
  }

 private:
  friend class CommonModifiers<Trivariate>;
  friend class SpatialModifiers<Trivariate>;
  friend class WindowedFactories<Trivariate>;

  /// Spatial interpolation configuration
  Spatial spatial_;
  /// Third axis interpolation configuration
  AxisConfig third_axis_;
  /// Common interpolation configuration
  Common common_;
};

/// @brief Windows interpolation configuration (4D only)
class Quadrivariate : public CommonModifiers<Quadrivariate>,
                      public SpatialModifiers<Quadrivariate>,
                      public WindowedFactories<Quadrivariate> {
 public:
  /// @brief Default constructor
  constexpr Quadrivariate() = default;

  /// @brief Constructor with spatial, third axis, fourth axis, and common
  /// configurations
  /// @param[in] spatial Spatial interpolation configuration
  /// @param[in] third_axis Third axis interpolation configuration
  /// @param[in] fourth_axis Fourth axis interpolation configuration
  /// @param[in] common Common interpolation configuration
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

  /// @brief Update third axis configuration
  /// @param[in] config New third axis configuration
  /// @return Updated QuadrivariateConfig instance
  [[nodiscard]] constexpr auto with_third_axis(this Quadrivariate self,
                                               const AxisConfig& config)
      -> Quadrivariate {
    self.third_axis_ = config;
    return self;
  }

  /// @brief Update fourth axis configuration
  /// @param[in] config New fourth axis configuration
  /// @return Updated QuadrivariateConfig instance
  [[nodiscard]] constexpr auto with_fourth_axis(this Quadrivariate self,
                                                const AxisConfig& config)
      -> Quadrivariate {
    self.fourth_axis_ = config;
    return self;
  }

 private:
  friend class CommonModifiers<Quadrivariate>;
  friend class SpatialModifiers<Quadrivariate>;
  friend class WindowedFactories<Quadrivariate>;

  /// Spatial interpolation configuration
  Spatial spatial_;
  /// Third axis interpolation configuration
  AxisConfig third_axis_;
  /// Fourth axis interpolation configuration
  AxisConfig fourth_axis_;
  /// Common interpolation configuration
  Common common_;
};

}  // namespace windowed
}  // namespace pyinterp::config
