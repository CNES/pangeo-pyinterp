// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>
#include <Eigen/Core>
#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/geometry/box.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/geodetic/point.hpp"

namespace pyinterp::geodetic {

// Defines a box made of two describing points.
template <typename T>
class Box2D : public boost::geometry::model::box<Point2D<T>> {
 public:
  /// @brief Default constructor
  Box2D() = default;

  /// @brief Constructor taking the minimum corner point and the maximum corner
  /// point.
  /// @param min_corner the minimum corner point
  /// @param max_corner the maximum corner point
  Box2D(const Point2D<T>& min_corner, const Point2D<T>& max_corner)
      : boost::geometry::model::box<Point2D<T>>(min_corner, max_corner) {}

  /// @brief Get a box that covers the entire Earth. In other words, a box that
  /// covers all positions, whatever they may be.
  ///
  // @return a box that covers the entire Earth
  static auto entire_earth() -> Box2D { return Box2D({-180, -90}, {180, 90}); }

  /// @brief Test if the given point is inside or on border of this instance
  ///
  /// @param pt Point to test
  //  @return True if the given point is inside or on border of this Box
  auto covered_by(const Point2D<T>& pt) const -> bool {
    return boost::geometry::covered_by(pt, *this);
  }

  /// @brief Test if the coordinates of the points provided are located inside
  /// or at the edge of this box.
  ///
  /// @param lon Longitudes coordinates in degrees to check
  /// @param lat Latitude coordinates in degrees to check
  /// @return Returns a vector containing a flag equal to 1 if the coordinate is
  /// located in the box or at the edge otherwise 0.
  [[nodiscard]] auto covered_by(const Eigen::Ref<const Eigen::VectorXd>& lon,
                                const Eigen::Ref<const Eigen::VectorXd>& lat,
                                const size_t num_threads) const
      -> pybind11::array_t<int8_t> {
    detail::check_eigen_shape("lon", lon, "lat", lat);
    auto size = lon.size();
    auto result =
        pybind11::array_t<int8_t>(pybind11::array::ShapeContainer{{size}});
    auto _result = result.template mutable_unchecked<1>();

    {
      pybind11::gil_scoped_release release;

      // Captures the detected exceptions in the calculation function
      // (only the last exception captured is kept)
      auto except = std::exception_ptr(nullptr);

      detail::dispatch(
          [&](size_t start, size_t end) {
            try {
              for (size_t ix = start; ix < end; ++ix) {
                _result(ix) = static_cast<int8_t>(
                    covered_by(Point2D<T>(lon(ix), lat(ix))));
              }
            } catch (...) {
              except = std::current_exception();
            }
          },
          size, num_threads);

      if (except != nullptr) {
        std::rethrow_exception(except);
      }
    }
    return result;
  }

  /// Converts a Box2D into a string with the same meaning as that of this
  /// instance.
  [[nodiscard]] auto to_string() const -> std::string {
    std::stringstream ss;
    ss << boost::geometry::dsv(*this);
    return ss.str();
  }

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(this->min_corner().getstate(),
                                this->max_corner().getstate());
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple& state) -> Box2D<T> {
    if (state.size() != 2) {
      throw std::runtime_error("invalid state");
    }
    return Box2D<T>(Point2D<T>::setstate(state[0].cast<pybind11::tuple>()),
                    Point2D<T>::setstate(state[1].cast<pybind11::tuple>()));
  }
};

}  // namespace pyinterp::geodetic

// BOOST specialization to accept pyinterp::geodectic::Box2D as a geometry
// entity
namespace boost::geometry::traits {

namespace pg = pyinterp::geodetic;

/// Box tag
template <typename T>
struct tag<pg::Box2D<T>> {
  using type = box_tag;
};

/// Type of a point
template <typename T>
struct point_type<pg::Box2D<T>> {
  using type = pg::Point2D<T>;
};

template <typename T, std::size_t Dimension>
struct indexed_access<pg::Box2D<T>, min_corner, Dimension> {
  /// get corner of box
  static inline auto get(pg::Box2D<T> const& box) -> double {
    return geometry::get<Dimension>(box.min_corner());
  }

  /// set corner of box
  static inline void set(pg::Box2D<T>& box,  // NOLINT
                         double const& value) {
    geometry::set<Dimension>(box.min_corner(), value);
  }
};

template <typename T, std::size_t Dimension>
struct indexed_access<pg::Box2D<T>, max_corner, Dimension> {
  /// get corner of box
  static inline auto get(pg::Box2D<T> const& box) -> double {
    return geometry::get<Dimension>(box.max_corner());
  }

  /// set corner of box
  static inline void set(pg::Box2D<T>& box,  // NOLINT
                         double const& value) {
    geometry::set<Dimension>(box.max_corner(), value);
  }
};

}  // namespace boost::geometry::traits
