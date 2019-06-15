#pragma once
#include <boost/geometry.hpp>
#include <cmath>
#include <tuple>

namespace pyinterp {
namespace detail {
namespace math {

/// Abstract class for bivariate interpolation
template <typename Point, typename T>
struct Bivariate {
  /// Performs the interpolation
  ///
  /// @param p Query point
  /// @param p00 Point of coordinate (x0, y0)
  /// @param p01 Point of coordinate (x1, y1)
  /// @param z00 Point value for the coordinate (x0, y0)
  /// @param z01 Point value for the coordinate (x0, y1)
  /// @param z10 Point value for the coordinate (x1, y0)
  /// @param z11 Point value for the coordinate (x1, y1)
  /// @return interpolated value at coordinate (x, y)
  virtual T evaluate(const Point& p, const Point& p0, const Point& p1,
                     const T& z00, const T& z01, const T& z10,
                     const T& z11) = 0;
};

/// Bilinear interpolation
template <typename Point, typename T>
struct Bilinear : public Bivariate<Point, T> {
  /// Performs the bilinear interpolation
  inline T evaluate(const Point& p, const Point& p0, const Point& p1,
                    const T& z00, const T& z01, const T& z10,
                    const T& z11) final {
    auto dx = boost::geometry::get<0>(p1) - boost::geometry::get<0>(p0);
    auto dy = boost::geometry::get<1>(p1) - boost::geometry::get<1>(p0);
    auto t = (boost::geometry::get<0>(p) - boost::geometry::get<0>(p0)) / dx;
    auto u = (boost::geometry::get<1>(p) - boost::geometry::get<1>(p0)) / dy;
    return (T(1) - t) * (T(1) - u) * z00 + t * (T(1) - u) * z10 +
           (T(1) - t) * u * z01 + t * u * z11;
  }
};

/// Inverse distance weigthing interpolation
///
/// @see https://en.wikipedia.org/wiki/Inverse_distance_weighting
///
template <typename Point, typename T>
struct InverseDistanceWeigthing : public Bivariate<Point, T> {
  /// Default constructor (p=2)
  InverseDistanceWeigthing() = default;

  /// Explicit definition of the parameter p.
  explicit InverseDistanceWeigthing(const int exp) : exp_(exp) {}

  /// Performs the interpolation
  inline T evaluate(const Point& p, const Point& p0, const Point& p1,
                    const T& z00, const T& z01, const T& z10,
                    const T& z11) final {
    auto distance = boost::geometry::distance(
        p, Point{boost::geometry::get<0>(p0), boost::geometry::get<1>(p0)});
    if (distance <= std::numeric_limits<T>::epsilon()) {
      return z00;
    }

    auto w = 1 / std::pow(distance, exp_);
    auto wu = z00 * w;

    distance = boost::geometry::distance(
        p, Point{boost::geometry::get<0>(p0), boost::geometry::get<1>(p1)});

    if (distance <= std::numeric_limits<T>::epsilon()) {
      return z01;
    }

    auto wi = 1 / std::pow(distance, exp_);
    w += wi;
    wu += z01 * wi;

    distance = boost::geometry::distance(
        p, Point{boost::geometry::get<0>(p1), boost::geometry::get<1>(p0)});

    if (distance <= std::numeric_limits<T>::epsilon()) {
      return z10;
    }

    wi = 1 / std::pow(distance, exp_);
    w += wi;
    wu += z10 * wi;

    distance = boost::geometry::distance(
        p, Point{boost::geometry::get<0>(p1), boost::geometry::get<1>(p1)});

    if (distance <= std::numeric_limits<T>::epsilon()) {
      return z11;
    }

    wi = 1 / std::pow(distance, exp_);
    w += wi;
    wu += z11 * wi;

    return wu / w;
  }

 private:
  int exp_{2};
};

/// Nearest interpolation
template <typename Point, typename T>
struct Nearest : public Bivariate<Point, T> {
  /// Performs the interpolation
  inline T evaluate(const Point& p, const Point& p0, const Point& p1,
                    const T& z00, const T& z01, const T& z10,
                    const T& z11) final {
    auto distance = boost::geometry::comparable_distance(
        p, Point{boost::geometry::get<0>(p0), boost::geometry::get<1>(p0)});
    auto result = std::make_tuple(distance, z00);

    distance = boost::geometry::comparable_distance(
        p, Point{boost::geometry::get<0>(p0), boost::geometry::get<1>(p1)});
    if (std::get<0>(result) > distance) {
      result = std::make_tuple(distance, z01);
    }

    distance = boost::geometry::comparable_distance(
        p, Point{boost::geometry::get<0>(p1), boost::geometry::get<1>(p0)});
    if (std::get<0>(result) > distance) {
      result = std::make_tuple(distance, z10);
    }

    distance = boost::geometry::comparable_distance(
        p, Point{boost::geometry::get<0>(p1), boost::geometry::get<1>(p1)});
    if (std::get<0>(result) > distance) {
      result = std::make_tuple(distance, z11);
    }
    return std::get<1>(result);
  }
};

}  // namespace math
}  // namespace detail
}  // namespace pyinterp