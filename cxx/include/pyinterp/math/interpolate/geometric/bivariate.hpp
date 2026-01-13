// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once
#include <algorithm>
#include <array>
#include <boost/geometry.hpp>
#include <cmath>
#include <concepts>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace pyinterp::math::interpolate::geometric {

/// Concept for numeric types suitable for interpolation
template <typename T>
concept Numeric = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

/// Concept for point types compatible with boost::geometry
template <typename P>
concept GeometryPoint = requires(P p) {
  { boost::geometry::get<0>(p) } -> std::convertible_to<typename P::value_type>;
  { boost::geometry::get<1>(p) } -> std::convertible_to<typename P::value_type>;
};

/// Abstract class for bivariate interpolation
/// @tparam Point Point type (must satisfy GeometryPoint concept)
/// @tparam T Numeric type for coordinates and values
template <template <class> class Point, typename T>
  requires Numeric<T>
struct Bivariate {
  /// Default constructor
  constexpr Bivariate() = default;

  /// Default destructor
  constexpr virtual ~Bivariate() = default;

  /// Copy constructor
  constexpr Bivariate(const Bivariate &) = default;

  /// Move constructor
  constexpr Bivariate(Bivariate &&) noexcept = default;

  /// Copy assignment operator
  constexpr auto operator=(const Bivariate &) -> Bivariate & = default;

  /// Move assignment operator
  constexpr auto operator=(Bivariate &&) noexcept -> Bivariate & = default;

  /// @brief Performs the interpolation
  ///
  /// @param[in] p Query point
  /// @param[in] p0 Point of coordinate (x0, y0)
  /// @param[in] p1 Point of coordinate (x1, y1)
  /// @param[in] q00 Point value for the coordinate (x0, y0)
  /// @param[in] q01 Point value for the coordinate (x0, y1)
  /// @param[in] q10 Point value for the coordinate (x1, y0)
  /// @param[in] q11 Point value for the coordinate (x1, y1)
  /// @return interpolated value at coordinate (x, y)
  [[nodiscard]] constexpr virtual auto evaluate(
      const Point<T> &p, const Point<T> &p0, const Point<T> &p1, const T &q00,
      const T &q01, const T &q10, const T &q11) const -> T = 0;
};

/// @brief Bilinear interpolation
///
/// Performs standard bilinear interpolation using the four corner values
/// of a rectangular grid cell.
/// @tparam Point Point type (must satisfy GeometryPoint concept)
/// @tparam T Numeric type for coordinates and values
template <template <class> class Point, typename T>
  requires Numeric<T>
struct Bilinear final : public Bivariate<Point, T> {
  /// Default constructor
  constexpr Bilinear() = default;

  /// Default destructor
  constexpr ~Bilinear() override = default;

  /// Copy constructor
  constexpr Bilinear(const Bilinear &) = default;

  /// Move constructor
  constexpr Bilinear(Bilinear &&) noexcept = default;

  /// Copy assignment operator
  constexpr auto operator=(const Bilinear &) -> Bilinear & = default;

  /// Move assignment operator
  constexpr auto operator=(Bilinear &&) noexcept -> Bilinear & = default;

  /// @brief Performs the bilinear interpolation
  ///
  /// Uses the standard bilinear formula:
  /// f(x,y) = (1-t)(1-u)q00 + t(1-u)q10 + (1-t)u*q01 + t*u*q11
  /// where t = (x-x0)/(x1-x0) and u = (y-y0)/(y1-y0)
  ///
  /// @param[in] p Query point
  /// @param[in] p0 Point of coordinate (x0, y0)
  /// @param[in] p1 Point of coordinate (x1, y1)
  /// @param[in] q00 Point value for the coordinate (x0, y0)
  /// @param[in] q01 Point value for the coordinate (x0, y1)
  /// @param[in] q10 Point value for the coordinate (x1, y0)
  /// @param[in] q11 Point value for the coordinate (x1, y1)
  /// @return interpolated value at coordinate (x, y)
  [[nodiscard]] constexpr auto evaluate(const Point<T> &p, const Point<T> &p0,
                                        const Point<T> &p1, const T &q00,
                                        const T &q01, const T &q10,
                                        const T &q11) const -> T override {
    const auto x = boost::geometry::get<0>(p);
    const auto y = boost::geometry::get<1>(p);
    const auto x0 = boost::geometry::get<0>(p0);
    const auto y0 = boost::geometry::get<1>(p0);
    const auto x1 = boost::geometry::get<0>(p1);
    const auto y1 = boost::geometry::get<1>(p1);

    const auto dx = x1 - x0;
    const auto dy = y1 - y0;
    const auto t = (x - x0) / dx;
    const auto u = (y - y0) / dy;

    // Compute using Horner-like scheme for better numerical stability
    const auto one_minus_t = T(1) - t;
    const auto one_minus_u = T(1) - u;

    return one_minus_t * (one_minus_u * q00 + u * q01) +
           t * (one_minus_u * q10 + u * q11);
  }
};

/// @brief Inverse distance weighting interpolation
///
/// Implements IDW interpolation with configurable power parameter.
/// Points closer to the query point have more influence on the result.
///
/// @see https://en.wikipedia.org/wiki/Inverse_distance_weighting
/// @tparam Point Point type (must satisfy GeometryPoint concept)
/// @tparam T Numeric type for coordinates and values
template <template <class> class Point, typename T>
  requires Numeric<T>
struct InverseDistanceWeighting final : public Bivariate<Point, T> {
  /// Default constructor (p=2, standard IDW)
  constexpr InverseDistanceWeighting() = default;

  /// @brief Explicit definition of the power parameter
  ///
  /// @param[in] exp Power parameter (typically 1-3, where 2 is standard)
  explicit constexpr InverseDistanceWeighting(int exp) : exp_(exp) {}

  /// Return the exponent used by this instance
  [[nodiscard]] constexpr auto exp() const noexcept -> int { return exp_; }

  /// Default destructor
  constexpr ~InverseDistanceWeighting() override = default;

  /// Copy constructor
  constexpr InverseDistanceWeighting(const InverseDistanceWeighting &) =
      default;

  /// Move constructor
  constexpr InverseDistanceWeighting(InverseDistanceWeighting &&) noexcept =
      default;

  /// Copy assignment operator
  constexpr auto operator=(const InverseDistanceWeighting &)
      -> InverseDistanceWeighting & = default;

  /// Move assignment operator
  constexpr auto operator=(InverseDistanceWeighting &&) noexcept
      -> InverseDistanceWeighting & = default;

  /// @brief Performs the interpolation using inverse distance weighting
  ///
  /// If the query point coincides with a data point (within epsilon),
  /// returns that data point's value directly.
  ///
  /// @param[in] p Query point
  /// @param[in] p0 Point of coordinate (x0, y0)
  /// @param[in] p1 Point of coordinate (x1, y1)
  /// @param[in] q00 Point value for the coordinate (x0, y0)
  /// @param[in] q01 Point value for the coordinate (x0, y1)
  /// @param[in] q10 Point value for the coordinate (x1, y0)
  /// @param[in] q11 Point value for the coordinate (x1, y1)
  /// @return interpolated value at coordinate (x, y)
  [[nodiscard]] auto evaluate(const Point<T> &p, const Point<T> &p0,
                              const Point<T> &p1, const T &q00, const T &q01,
                              const T &q10, const T &q11) const -> T override {
    const std::array corners = {
        std::pair{
            Point<T>{boost::geometry::get<0>(p0), boost::geometry::get<1>(p0)},
            q00},
        std::pair{
            Point<T>{boost::geometry::get<0>(p0), boost::geometry::get<1>(p1)},
            q01},
        std::pair{
            Point<T>{boost::geometry::get<0>(p1), boost::geometry::get<1>(p0)},
            q10},
        std::pair{
            Point<T>{boost::geometry::get<0>(p1), boost::geometry::get<1>(p1)},
            q11}};

    constexpr auto epsilon = std::numeric_limits<T>::epsilon();
    constexpr auto epsilon_squared = epsilon * epsilon;
    T weight_sum = T(0);
    T weighted_value_sum = T(0);

    for (const auto &corner : corners) {
      const auto &corner_point = corner.first;
      const auto corner_value = corner.second;

      const auto dist_squared =
          boost::geometry::comparable_distance(p, corner_point);

      // Check for coincident point using squared distance
      if (dist_squared <= epsilon_squared) {
        return corner_value;
      }

      // Compute weight: 1 / distance^exp = 1 / (sqrt(dist_squared))^exp
      // = 1 / dist_squared^(exp/2)
      const auto weight =
          static_cast<T>(1 / std::pow(dist_squared, exp_ * 0.5));
      weight_sum += weight;
      weighted_value_sum += corner_value * weight;
    }

    return weighted_value_sum / weight_sum;
  }

 private:
  int exp_{2};  ///< Power parameter for IDW (default: 2)
};

/// @brief Nearest neighbor interpolation
///
/// @tparam Point Point type (must satisfy GeometryPoint concept)
/// @tparam T Numeric type for coordinates and values
template <template <class> class Point, typename T>
  requires Numeric<T>
struct Nearest final : public Bivariate<Point, T> {
  /// Default constructor
  constexpr Nearest() = default;

  /// Default destructor
  constexpr ~Nearest() override = default;

  /// Copy constructor
  constexpr Nearest(const Nearest &) = default;

  /// Move constructor
  constexpr Nearest(Nearest &&) noexcept = default;

  /// Copy assignment operator
  constexpr auto operator=(const Nearest &) -> Nearest & = default;

  /// Move assignment operator
  constexpr auto operator=(Nearest &&) noexcept -> Nearest & = default;

  /// @brief Performs nearest neighbor interpolation
  ///
  /// Finds the corner point with minimum distance to the query point
  /// and returns its value.
  ///
  /// @param[in] p Query point
  /// @param[in] p0 Point of coordinate (x0, y0)
  /// @param[in] p1 Point of coordinate (x1, y1)
  /// @param[in] q00 Point value for the coordinate (x0, y0)
  /// @param[in] q01 Point value for the coordinate (x0, y1)
  /// @param[in] q10 Point value for the coordinate (x1, y0)
  /// @param[in] q11 Point value for the coordinate (x1, y1)
  /// @return interpolated value at coordinate (x, y)
  [[nodiscard]] auto evaluate(const Point<T> &p, const Point<T> &p0,
                              const Point<T> &p1, const T &q00, const T &q01,
                              const T &q10, const T &q11) const -> T override {
    // Define corner points with their values
    const std::array corners = {
        std::pair{
            Point<T>{boost::geometry::get<0>(p0), boost::geometry::get<1>(p0)},
            q00},
        std::pair{
            Point<T>{boost::geometry::get<0>(p0), boost::geometry::get<1>(p1)},
            q01},
        std::pair{
            Point<T>{boost::geometry::get<0>(p1), boost::geometry::get<1>(p0)},
            q10},
        std::pair{
            Point<T>{boost::geometry::get<0>(p1), boost::geometry::get<1>(p1)},
            q11}};

    // Use comparable_distance (avoids sqrt) and find minimum
    auto min_corner =
        std::ranges::min(corners, [&p](const auto &a, const auto &b) {
          return boost::geometry::comparable_distance(p, a.first) <
                 boost::geometry::comparable_distance(p, b.first);
        });

    return min_corner.second;
  }
};

/// @brief Helper function to create points from coordinates
///
/// @param[in] x X coordinate
/// @param[in] y Y coordinate
/// @return Point with the given coordinates
template <template <class> class Point, typename T>
[[nodiscard]] constexpr auto make_point(T x, T y) -> Point<T> {
  return Point<T>{x, y};
}

/// Interpolation method selection helper
enum class InterpolationMethod {
  kBilinear,                  ///< Bilinear interpolation
  kInverseDistanceWeighting,  ///< IDW interpolation
  kNearest                    ///< Nearest neighbor
};

/// @brief Factory function to create interpolation method
///
/// @tparam Point Point type template
/// @tparam T Numeric type
/// @param[in] method Interpolation method to use
/// @param[in] idw_power Power parameter for IDW (ignored for other methods)
/// @return Unique pointer to the interpolation object
template <template <class> class Point, typename T>
  requires Numeric<T>
[[nodiscard]] auto make_interpolator(InterpolationMethod method,
                                     int idw_power = 2)
    -> std::unique_ptr<Bivariate<Point, T>> {
  switch (method) {
    case InterpolationMethod::kBilinear:
      return std::make_unique<Bilinear<Point, T>>();
    case InterpolationMethod::kInverseDistanceWeighting:
      return std::make_unique<InverseDistanceWeighting<Point, T>>(idw_power);
    case InterpolationMethod::kNearest:
      return std::make_unique<Nearest<Point, T>>();
  }
  std::unreachable();
}

}  // namespace pyinterp::math::interpolate::geometric
