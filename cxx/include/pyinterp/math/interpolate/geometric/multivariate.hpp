// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <functional>
#include <type_traits>
#include <utility>

#include "pyinterp/math/interpolate/geometric/bivariate.hpp"
#include "pyinterp/math/interpolate/linear.hpp"
#include "pyinterp/math/interpolate/nearest.hpp"

namespace pyinterp::math::interpolate::geometric {

/// Concept for axis coordinate types
template <typename T>
concept AxisCoordinate = std::is_arithmetic_v<T>;

/// Method for interpolation along non-spatial axes
enum class AxisMethod : uint8_t {
  kLinear,  ///< Linear interpolation
  kNearest  ///< Nearest neighbor interpolation
};

/// Type alias for 1D axis interpolation function
template <typename Axis, typename T>
using AxisInterpolator = std::function<T(Axis, Axis, Axis, T, T)>;

/// @brief Get the interpolation function for any axis
///
/// This factory is used by both trivariate and quadrivariate
///
/// @tparam Axis Axis coordinate type
/// @tparam T Numeric type for values
template <typename Axis, typename T>
  requires AxisCoordinate<Axis> && Numeric<T>
[[nodiscard]] constexpr auto get_axis_interpolator(AxisMethod method)
    -> AxisInterpolator<Axis, T> {
  switch (method) {
    case AxisMethod::kLinear:
      return &linear<Axis, T>;
    case AxisMethod::kNearest:
      return &nearest<Axis, T>;
  }
  std::unreachable();
}

// ============================================================================
// TRIVARIATE INTERPOLATION (3D: x, y, z)
// ============================================================================

/// @brief Container for a 2D spatial point with one additional axis
/// @tparam Point Point type template
/// @tparam T Numeric type for spatial coordinates
/// @tparam Z Type for the additional axis coordinate
template <template <class> class Point, typename T, typename Z = T>
  requires Numeric<T> && AxisCoordinate<Z>
struct SpatialPoint3D {
  Point<T> spatial;  ///< 2D spatial coordinates (x, y)
  Z third_axis;      ///< Third-axis coordinate (z: depth, time, etc.)

  /// @brief Constructor from spatial point and third-axis coordinate
  /// @param[in] spatial 2D spatial point
  /// @param[in] z Third-axis coordinate
  constexpr SpatialPoint3D(Point<T> spatial, Z z)
      : spatial(std::move(spatial)), third_axis(z) {}

  /// @brief Default constructor
  constexpr SpatialPoint3D() = default;
};

/// @brief Data cube: 8 corner values of a 3D grid cell
///
/// Naming: q[x][y][z] where 0=lower, 1=upper
template <typename T>
  requires Numeric<T>
struct DataCube {
  T q000;  ///< Value at corner (x0, y0, z0)
  T q010;  ///< Value at corner (x0, y1, z0)
  T q100;  ///< Value at corner (x1, y0, z0)
  T q110;  ///< Value at corner (x1, y1, z0)
  T q001;  ///< Value at corner (x0, y0, z1)
  T q011;  ///< Value at corner (x0, y1, z1)
  T q101;  ///< Value at corner (x1, y0, z1)
  T q111;  ///< Value at corner (x1, y1, z1)

  /// @brief Constructor from corner values
  /// @param[in] v000 Value at corner (x0, y0, z0)
  /// @param[in] v010 Value at corner (x0, y1, z0)
  /// @param[in] v100 Value at corner (x1, y0, z0)
  /// @param[in] v110 Value at corner (x1, y1, z0)
  /// @param[in] v001 Value at corner (x0, y0, z1)
  /// @param[in] v011 Value at corner (x0, y1, z1)
  /// @param[in] v101 Value at corner (x1, y0, z1)
  /// @param[in] v111 Value at corner (x1, y1, z1)
  constexpr DataCube(T v000, T v010, T v100, T v110, T v001, T v011, T v101,
                     T v111)
      : q000(v000),
        q010(v010),
        q100(v100),
        q110(v110),
        q001(v001),
        q011(v011),
        q101(v101),
        q111(v111) {}

  /// @brief Default constructor
  constexpr DataCube() = default;
};

/// @brief Trivariate interpolation: 2D spatial + 1D third-axis
///
/// Process:
/// 1. Spatial interpolation at z0 level
/// 2. Spatial interpolation at z1 level
/// 3. Interpolate along z-axis between the two results
///
/// @tparam Point Point type template
/// @tparam T Numeric type for spatial coordinates and values
/// @tparam Z Type for the third-axis coordinate
/// @param[in] query Query point (2D spatial + 1D third-axis)
/// @param[in] bounds_lower Lower bounds of the grid cell containing the query
/// point
/// @param[in] bounds_upper Upper bounds of the grid cell containing the query
/// point
/// @param[in] data Data cube with the 8 corner values of the grid cell
/// @param[in] spatial_interpolator Bivariate spatial interpolator
/// @param[in] axis_interpolator 1D axis interpolator
/// @return Interpolated value at the query point
template <template <class> class Point, typename T, typename Z = T>
  requires Numeric<T> && AxisCoordinate<Z>
[[nodiscard]] auto trivariate(const SpatialPoint3D<Point, T, Z> &query,
                              const SpatialPoint3D<Point, T, Z> &bounds_lower,
                              const SpatialPoint3D<Point, T, Z> &bounds_upper,
                              const DataCube<T> &data,
                              const Bivariate<Point, T> &spatial_interpolator,
                              const AxisInterpolator<Z, T> &axis_interpolator)
    -> T {
  // Step 1: Interpolate spatially at z0 level
  const auto z0 = spatial_interpolator.evaluate(
      query.spatial, bounds_lower.spatial, bounds_upper.spatial, data.q000,
      data.q010, data.q100, data.q110);

  // Step 2: Interpolate spatially at z1 level
  const auto z1 = spatial_interpolator.evaluate(
      query.spatial, bounds_lower.spatial, bounds_upper.spatial, data.q001,
      data.q011, data.q101, data.q111);

  // Step 3: Interpolate along z-axis
  return axis_interpolator(query.third_axis, bounds_lower.third_axis,
                           bounds_upper.third_axis, z0, z1);
}

// ============================================================================
// QUADRIVARIATE INTERPOLATION (4D: x, y, z, u)
// ============================================================================

/// @brief Container for a 2D spatial point with two additional axes
/// @tparam Point Point type template
/// @tparam T Numeric type for spatial coordinates
/// @tparam Z Type for the third-axis coordinate
/// @tparam U Type for the fourth-axis coordinate
template <template <class> class Point, typename T, typename Z = T,
          typename U = T>
  requires Numeric<T> && AxisCoordinate<Z> && AxisCoordinate<U>
struct SpatialPoint4D {
  Point<T> spatial;  ///< 2D spatial coordinates (x, y)
  Z z_axis;          ///< Third-axis coordinate (z)
  U u_axis;          ///< Fourth-axis coordinate (u)

  /// @brief Constructor from spatial point and additional axis coordinates
  /// @param[in] spatial 2D spatial point
  /// @param[in] z Third-axis coordinate
  /// @param[in] u Fourth-axis coordinate
  constexpr SpatialPoint4D(Point<T> spatial, Z z, U u)
      : spatial(std::move(spatial)), z_axis(z), u_axis(u) {}

  /// @brief Default constructor
  constexpr SpatialPoint4D() = default;
};

/// @brief Data hypercube: 16 corner values of a 4D grid cell
///
/// Naming: q[x][y][z][u] where 0=lower, 1=upper
template <typename T>
  requires Numeric<T>
struct DataHypercube {
  T q0000;  ///< Value at (x0, y0, z0, u0)
  T q0100;  ///< Value at (x0, y1, z0, u0)
  T q1000;  ///< Value at (x1, y0, z0, u0)
  T q1100;  ///< Value at (x1, y1, z0, u0)
  T q0010;  ///< Value at (x0, y0, z1, u0)
  T q0110;  ///< Value at (x0, y1, z1, u0)
  T q1010;  ///< Value at (x1, y0, z1, u0)
  T q1110;  ///< Value at (x1, y1, z1, u0)
  T q0001;  ///< Value at (x0, y0, z0, u1)
  T q0101;  ///< Value at (x0, y1, z0, u1)
  T q1001;  ///< Value at (x1, y0, z0, u1)
  T q1101;  ///< Value at (x1, y1, z0, u1)
  T q0011;  ///< Value at (x0, y0, z1, u1)
  T q0111;  ///< Value at (x0, y1, z1, u1)
  T q1011;  ///< Value at (x1, y0, z1, u1)
  T q1111;  ///< Value at (x1, y1, z1, u1)

  /// @brief Constructor from corner values
  /// @param[in] v0000 Value at (x0, y0, z0, u0)
  /// @param[in] v0100 Value at (x0, y1, z0, u0)
  /// @param[in] v1000 Value at (x1, y0, z0, u0)
  /// @param[in] v1100 Value at (x1, y1, z0, u0)
  /// @param[in] v0010 Value at (x0, y0, z1, u0)
  /// @param[in] v0110 Value at (x0, y1, z1, u0)
  /// @param[in] v1010 Value at (x1, y0, z1, u0)
  /// @param[in] v1110 Value at (x1, y1, z1, u0)
  /// @param[in] v0001 Value at (x0, y0, z0, u1)
  /// @param[in] v0101 Value at (x0, y1, z0, u1)
  /// @param[in] v1001 Value at (x1, y0, z0, u1)
  /// @param[in] v1101 Value at (x1, y1, z0, u1)
  /// @param[in] v0011 Value at (x0, y0, z1, u1)
  /// @param[in] v0111 Value at (x0, y1, z1, u1)
  /// @param[in] v1011 Value at (x1, y0, z1, u1)
  /// @param[in] v1111 Value at (x1, y1, z1, u1)
  constexpr DataHypercube(T v0000, T v0100, T v1000, T v1100, T v0010, T v0110,
                          T v1010, T v1110, T v0001, T v0101, T v1001, T v1101,
                          T v0011, T v0111, T v1011, T v1111)
      : q0000(v0000),
        q0100(v0100),
        q1000(v1000),
        q1100(v1100),
        q0010(v0010),
        q0110(v0110),
        q1010(v1010),
        q1110(v1110),
        q0001(v0001),
        q0101(v0101),
        q1001(v1001),
        q1101(v1101),
        q0011(v0011),
        q0111(v0111),
        q1011(v1011),
        q1111(v1111) {}

  /// @brief Default constructor
  constexpr DataHypercube() = default;
};

/// @brief Quadrivariate interpolation: 2D spatial + 2D additional axes
///
/// Process
/// 1. Apply trivariate interpolation at u0 level (spatial + z)
/// 2. Apply trivariate interpolation at u1 level (spatial + z)
/// 3. Interpolate along u-axis between the two results
///
/// @tparam Point Point type template
/// @tparam T Numeric type for spatial coordinates and values
/// @tparam Z Type for the third-axis coordinate
/// @tparam U Type for the fourth-axis coordinate
/// @param[in] query Query point (2D spatial + 2D additional axes)
/// @param[in] bounds_lower Lower bounds of the grid cell containing the query
/// point
/// @param[in] bounds_upper Upper bounds of the grid cell containing the query
/// point
/// @param[in] data Data hypercube with the 16 corner values of the grid cell
/// @param[in] spatial_interpolator Bivariate spatial interpolator
/// @param[in] z_axis_interpolator 1D interpolator for the third axis
/// @param[in] u_axis_interpolator 1D interpolator for the fourth axis
/// @return Interpolated value at the query point
template <template <class> class Point, typename T, typename Z = T,
          typename U = T>
  requires Numeric<T> && AxisCoordinate<Z> && AxisCoordinate<U>
[[nodiscard]] auto quadrivariate(
    const SpatialPoint4D<Point, T, Z, U> &query,
    const SpatialPoint4D<Point, T, Z, U> &bounds_lower,
    const SpatialPoint4D<Point, T, Z, U> &bounds_upper,
    const DataHypercube<T> &data,
    const Bivariate<Point, T> &spatial_interpolator,
    const AxisInterpolator<Z, T> &z_axis_interpolator,
    const AxisInterpolator<U, T> &u_axis_interpolator) -> T {
  // At u=0: Interpolate in (x, y, z) space
  const auto z0_u0 = spatial_interpolator.evaluate(
      query.spatial, bounds_lower.spatial, bounds_upper.spatial, data.q0000,
      data.q0100, data.q1000, data.q1100);

  const auto z1_u0 = spatial_interpolator.evaluate(
      query.spatial, bounds_lower.spatial, bounds_upper.spatial, data.q0010,
      data.q0110, data.q1010, data.q1110);

  const auto u0 = z_axis_interpolator(query.z_axis, bounds_lower.z_axis,
                                      bounds_upper.z_axis, z0_u0, z1_u0);

  // At u=1: Interpolate in (x, y, z) space
  const auto z0_u1 = spatial_interpolator.evaluate(
      query.spatial, bounds_lower.spatial, bounds_upper.spatial, data.q0001,
      data.q0101, data.q1001, data.q1101);

  const auto z1_u1 = spatial_interpolator.evaluate(
      query.spatial, bounds_lower.spatial, bounds_upper.spatial, data.q0011,
      data.q0111, data.q1011, data.q1111);

  const auto u1 = z_axis_interpolator(query.z_axis, bounds_lower.z_axis,
                                      bounds_upper.z_axis, z0_u1, z1_u1);

  // Final u-axis interpolation
  return u_axis_interpolator(query.u_axis, bounds_lower.u_axis,
                             bounds_upper.u_axis, u0, u1);
}

}  // namespace pyinterp::math::interpolate::geometric
