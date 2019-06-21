#pragma once
#include <boost/geometry.hpp>
#include <cmath>
#include <tuple>

namespace pyinterp {
namespace detail {
namespace math {

/// Abstract class for bivariate interpolation
template <template <class> class Point, typename T>
struct Bivariate {
  /// Destructor
  virtual ~Bivariate() = default;

  /// Performs the interpolation
  ///
  /// @param p Query point
  /// @param p0 Point of coordinate (x0, y0)
  /// @param p1 Point of coordinate (x1, y1)
  /// @param q00 Point value for the coordinate (x0, y0)
  /// @param q01 Point value for the coordinate (x0, y1)
  /// @param q10 Point value for the coordinate (x1, y0)
  /// @param q11 Point value for the coordinate (x1, y1)
  /// @return interpolated value at coordinate (x, y)
  virtual T evaluate(const Point<T>& p, const Point<T>& p0, const Point<T>& p1,
                     const T& q00, const T& q01, const T& q10,
                     const T& q11) const = 0;
};

/// Bilinear interpolation
template <template <class> class Point, typename T>
struct Bilinear : public Bivariate<Point, T> {
  /// Destructor
  virtual ~Bilinear() = default;

  /// Performs the bilinear interpolation
  inline T evaluate(const Point<T>& p, const Point<T>& p0, const Point<T>& p1,
                    const T& q00, const T& q01, const T& q10,
                    const T& q11) const final {
    auto dx = boost::geometry::get<0>(p1) - boost::geometry::get<0>(p0);
    auto dy = boost::geometry::get<1>(p1) - boost::geometry::get<1>(p0);
    auto t = (boost::geometry::get<0>(p) - boost::geometry::get<0>(p0)) / dx;
    auto u = (boost::geometry::get<1>(p) - boost::geometry::get<1>(p0)) / dy;
    return (T(1) - t) * (T(1) - u) * q00 + t * (T(1) - u) * q10 +
           (T(1) - t) * u * q01 + t * u * q11;
  }
};

/// Inverse distance weigthing interpolation
///
/// @see https://en.wikipedia.org/wiki/Inverse_distance_weighting
///
template <template <class> class Point, typename T>
struct InverseDistanceWeighting : public Bivariate<Point, T> {
  /// Default constructor (p=2)
  InverseDistanceWeighting() = default;

  /// Explicit definition of the parameter p.
  explicit InverseDistanceWeighting(const int exp) : exp_(exp) {}

  /// Return the exponent used by this instance
  inline int exp() const noexcept { return exp_; }

  /// Destructor
  virtual ~InverseDistanceWeighting() = default;

  /// Performs the interpolation
  inline T evaluate(const Point<T>& p, const Point<T>& p0, const Point<T>& p1,
                    const T& q00, const T& q01, const T& q10,
                    const T& q11) const final {
    auto distance = boost::geometry::distance(
        p, Point<T>{boost::geometry::get<0>(p0), boost::geometry::get<1>(p0)});
    if (distance <= std::numeric_limits<T>::epsilon()) {
      return q00;
    }

    auto w = 1 / std::pow(distance, exp_);
    auto wu = q00 * w;

    distance = boost::geometry::distance(
        p, Point<T>{boost::geometry::get<0>(p0), boost::geometry::get<1>(p1)});

    if (distance <= std::numeric_limits<T>::epsilon()) {
      return q01;
    }

    auto wi = 1 / std::pow(distance, exp_);
    w += wi;
    wu += q01 * wi;

    distance = boost::geometry::distance(
        p, Point<T>{boost::geometry::get<0>(p1), boost::geometry::get<1>(p0)});

    if (distance <= std::numeric_limits<T>::epsilon()) {
      return q10;
    }

    wi = 1 / std::pow(distance, exp_);
    w += wi;
    wu += q10 * wi;

    distance = boost::geometry::distance(
        p, Point<T>{boost::geometry::get<0>(p1), boost::geometry::get<1>(p1)});

    if (distance <= std::numeric_limits<T>::epsilon()) {
      return q11;
    }

    wi = 1 / std::pow(distance, exp_);
    w += wi;
    wu += q11 * wi;

    return wu / w;
  }

 private:
  int exp_{2};
};

/// Nearest interpolation
template <template <class> class Point, typename T>
struct Nearest : public Bivariate<Point, T> {
  /// Performs the interpolation
  inline T evaluate(const Point<T>& p, const Point<T>& p0, const Point<T>& p1,
                    const T& q00, const T& q01, const T& q10,
                    const T& q11) const final {
    auto distance = boost::geometry::comparable_distance(
        p, Point<T>{boost::geometry::get<0>(p0), boost::geometry::get<1>(p0)});
    auto result = std::make_tuple(distance, q00);

    distance = boost::geometry::comparable_distance(
        p, Point<T>{boost::geometry::get<0>(p0), boost::geometry::get<1>(p1)});
    if (std::get<0>(result) > distance) {
      result = std::make_tuple(distance, q01);
    }

    distance = boost::geometry::comparable_distance(
        p, Point<T>{boost::geometry::get<0>(p1), boost::geometry::get<1>(p0)});
    if (std::get<0>(result) > distance) {
      result = std::make_tuple(distance, q10);
    }

    distance = boost::geometry::comparable_distance(
        p, Point<T>{boost::geometry::get<0>(p1), boost::geometry::get<1>(p1)});
    if (std::get<0>(result) > distance) {
      result = std::make_tuple(distance, q11);
    }
    return std::get<1>(result);
  }
};

}  // namespace math
}  // namespace detail
}  // namespace pyinterp