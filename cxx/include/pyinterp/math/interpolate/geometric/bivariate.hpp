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
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
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

  /// @brief Interpolates several layers of values sharing the same cell
  ///
  /// The 3D and 4D interpolations evaluate the same spatial cell for each
  /// level of their other axes. Methods whose cost lies in the geometry
  /// (distances to the corners) override this to compute it once.
  ///
  /// @param[in] p Query point
  /// @param[in] p0 Point of coordinate (x0, y0)
  /// @param[in] p1 Point of coordinate (x1, y1)
  /// @param[in] layers Corner values of each layer: {q00, q01, q10, q11}
  /// @param[out] result Interpolated value of each layer
  constexpr virtual void evaluate_layers(
      const Point<T> &p, const Point<T> &p0, const Point<T> &p1,
      std::span<const std::array<T, 4>> layers, std::span<T> result) const {
    for (size_t ix = 0; ix < layers.size(); ++ix) {
      const auto &q = layers[ix];
      result[ix] = evaluate(p, p0, p1, q[0], q[1], q[2], q[3]);
    }
  }
};

/// @brief Comparable distances from a point to the corners of a cell
/// @param[in] p Query point
/// @param[in] p0 Point of coordinate (x0, y0)
/// @param[in] p1 Point of coordinate (x1, y1)
/// @return Distances to (x0, y0), (x0, y1), (x1, y0) and (x1, y1)
template <template <class> class Point, typename T>
[[nodiscard]] auto corner_distances(const Point<T> &p, const Point<T> &p0,
                                    const Point<T> &p1) {
  const auto x0 = boost::geometry::get<0>(p0);
  const auto y0 = boost::geometry::get<1>(p0);
  const auto x1 = boost::geometry::get<0>(p1);
  const auto y1 = boost::geometry::get<1>(p1);
  return std::array{boost::geometry::comparable_distance(p, Point<T>{x0, y0}),
                    boost::geometry::comparable_distance(p, Point<T>{x0, y1}),
                    boost::geometry::comparable_distance(p, Point<T>{x1, y0}),
                    boost::geometry::comparable_distance(p, Point<T>{x1, y1})};
}

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

    return combine(weight(x, x0, x1), weight(y, y0, y1), q00, q01, q10, q11);
  }

  /// @brief Relative position of a coordinate in its cell
  ///
  /// @param[in] x Coordinate
  /// @param[in] x0 Lower bound of the cell
  /// @param[in] x1 Upper bound of the cell
  /// @return (x - x0) / (x1 - x0)
  [[nodiscard]] static constexpr auto weight(const T &x, const T &x0,
                                             const T &x1) -> T {
    const auto dx = x1 - x0;
    return (x - x0) / dx;
  }

  /// @brief Combines the corner values from the relative positions
  ///
  /// The interpolation of a grid computes the relative positions once per
  /// axis and calls this function for each point, which gives the same
  /// result as evaluate.
  ///
  /// @param[in] t Relative position along x
  /// @param[in] u Relative position along y
  /// @param[in] q00 Point value for the coordinate (x0, y0)
  /// @param[in] q01 Point value for the coordinate (x0, y1)
  /// @param[in] q10 Point value for the coordinate (x1, y0)
  /// @param[in] q11 Point value for the coordinate (x1, y1)
  /// @return interpolated value
  [[nodiscard]] static constexpr auto combine(const T &t, const T &u,
                                              const T &q00, const T &q01,
                                              const T &q10, const T &q11) -> T {
    // Compute using Horner-like scheme for better numerical stability
    const auto one_minus_t = T(1) - t;
    const auto one_minus_u = T(1) - u;

    return one_minus_t * (one_minus_u * q00 + u * q01) +
           t * (one_minus_u * q10 + u * q11);
  }

  /// @brief Interpolates several layers without a virtual call per layer
  constexpr void evaluate_layers(const Point<T> &p, const Point<T> &p0,
                                 const Point<T> &p1,
                                 std::span<const std::array<T, 4>> layers,
                                 std::span<T> result) const override {
    for (size_t ix = 0; ix < layers.size(); ++ix) {
      const auto &q = layers[ix];
      result[ix] = Bilinear::evaluate(p, p0, p1, q[0], q[1], q[2], q[3]);
    }
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
    const auto weights = corner_weights(p, p0, p1);
    return apply(weights, {q00, q01, q10, q11});
  }

  /// @brief Interpolates several layers, computing the weights once
  void evaluate_layers(const Point<T> &p, const Point<T> &p0,
                       const Point<T> &p1,
                       std::span<const std::array<T, 4>> layers,
                       std::span<T> result) const override {
    const auto weights = corner_weights(p, p0, p1);
    for (size_t ix = 0; ix < layers.size(); ++ix) {
      result[ix] = apply(weights, layers[ix]);
    }
  }

 private:
  /// Weights of the corners, or the index of the corner the query point
  /// coincides with
  struct Weights {
    std::array<T, 4> values;
    int coincident;
  };

  /// @brief Computes the weights of the corners
  [[nodiscard]] auto corner_weights(const Point<T> &p, const Point<T> &p0,
                                    const Point<T> &p1) const -> Weights {
    constexpr auto epsilon = std::numeric_limits<T>::epsilon();
    constexpr auto epsilon_squared = epsilon * epsilon;

    auto result = Weights{.values = {}, .coincident = -1};
    const auto distances = corner_distances<Point, T>(p, p0, p1);
    for (size_t ix = 0; ix < distances.size(); ++ix) {
      const auto dist_squared = distances[ix];
      // Check for coincident point using squared distance
      if (dist_squared <= epsilon_squared) {
        result.coincident = static_cast<int>(ix);
        return result;
      }
      // Compute weight: 1 / distance^exp = 1 / (sqrt(dist_squared))^exp
      // = 1 / dist_squared^(exp/2). pow(x, 1) is x, so the common exp = 2
      // skips the call with the same result.
      result.values[ix] = static_cast<T>(
          1 / (exp_ == 2 ? dist_squared : std::pow(dist_squared, exp_ * 0.5)));
    }
    return result;
  }

  /// @brief Applies the weights to the corner values
  [[nodiscard]] static auto apply(const Weights &weights,
                                  const std::array<T, 4> &values) -> T {
    if (weights.coincident != -1) {
      return values[weights.coincident];
    }
    T weight_sum = T(0);
    T weighted_value_sum = T(0);
    for (size_t ix = 0; ix < values.size(); ++ix) {
      weight_sum += weights.values[ix];
      weighted_value_sum += values[ix] * weights.values[ix];
    }
    return weighted_value_sum / weight_sum;
  }

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
    return std::array{q00, q01, q10, q11}[nearest_corner(p, p0, p1)];
  }

  /// @brief Interpolates several layers, searching the nearest corner once
  void evaluate_layers(const Point<T> &p, const Point<T> &p0,
                       const Point<T> &p1,
                       std::span<const std::array<T, 4>> layers,
                       std::span<T> result) const override {
    const auto corner = nearest_corner(p, p0, p1);
    for (size_t ix = 0; ix < layers.size(); ++ix) {
      result[ix] = layers[ix][corner];
    }
  }

 private:
  /// @brief Index of the nearest corner; the first one in case of a tie
  [[nodiscard]] static auto nearest_corner(const Point<T> &p,
                                           const Point<T> &p0,
                                           const Point<T> &p1) -> size_t {
    const auto distances = corner_distances<Point, T>(p, p0, p1);
    size_t result = 0;
    for (size_t ix = 1; ix < distances.size(); ++ix) {
      if (distances[ix] < distances[result]) {
        result = ix;
      }
    }
    return result;
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
enum class InterpolationMethod : uint8_t {
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
