// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <boost/geometry.hpp>

#include "pyinterp/geodetic/algorithm.hpp"
#include "pyinterp/geodetic/point.hpp"

namespace pyinterp::geodetic {

class Polygon : public boost::geometry::model::polygon<Point> {
 public:
  using boost::geometry::model::polygon<Point>::polygon;

  /// Create a new instance from Python
  Polygon(const pybind11::list& outer, const pybind11::list& inners) {
    try {
      for (const auto item : outer) {
        auto point = item.cast<geodetic::Point>();
        boost::geometry::append(this->outer(), point);
      }
    } catch (const pybind11::cast_error&) {
      throw std::invalid_argument(
          "outer must be a list of pyinterp.geodetic.Point");
    }
    if (!inners.empty()) {
      try {
        auto index = 0;
        this->inners().resize(inners.size());
        for (const auto inner : inners) {
          auto points = inner.cast<pybind11::list>();
          for (const auto item : points) {
            auto point = item.cast<geodetic::Point>();
            boost::geometry::append(this->inners()[index], point);
          }
          ++index;
        }
      } catch (const pybind11::cast_error&) {
        throw std::invalid_argument(
            "inners must be a list of "
            "list of pyinterp.geodetic.Point");
      }
    }
  }

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    auto inners = pybind11::list();
    auto outer = pybind11::list();

    for (const auto& item : this->outer()) {
      outer.append(item);
    }

    for (const auto& inner : this->inners()) {
      auto buffer = pybind11::list();
      for (const auto& item : inner) {
        buffer.append(item);
      }
      inners.append(buffer);
    }
    return pybind11::make_tuple(outer, inners);
  }

  /// Calculate the area
  [[nodiscard]] auto area(const std::optional<System>& wgs) const -> double {
    return geodetic::area(*this, wgs);
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple& state) -> Polygon {
    if (state.size() != 2) {
      throw std::runtime_error("invalid state");
    }
    return Polygon(state[0].cast<pybind11::list>(),
                   state[1].cast<pybind11::list>());
  }

  /// @brief Test if the given point is inside or on border of this instance
  ///
  /// @param pt Point to test
  //  @return True if the given point is inside or on border of this Polygon
  [[nodiscard]] auto covered_by(const Point& point) const -> bool {
    return boost::geometry::covered_by(point, *this);
  }

  /// @brief Test if the coordinates of the points provided are located inside
  /// or at the edge of this Polygon.
  ///
  /// @param lon Longitudes coordinates in degrees to check
  /// @param lat Latitude coordinates in degrees to check
  /// @return Returns a vector containing a flag equal to 1 if the coordinate is
  /// located in the Polygon or at the edge otherwise 0.
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
                _result(ix) =
                    static_cast<int8_t>(covered_by({lon(ix), lat(ix)}));
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

  /// Converts a Polygon into a string with the same meaning as that of this
  /// instance.
  [[nodiscard]] auto to_string() const -> std::string {
    std::stringstream ss;
    ss << boost::geometry::dsv(*this);
    return ss.str();
  }
};

}  // namespace pyinterp::geodetic

namespace boost::geometry::traits {
namespace pg = pyinterp::geodetic;

template <>
struct tag<pg::Polygon> {
  using type = polygon_tag;
};
template <>
struct ring_const_type<pg::Polygon> {
  using type = const model::polygon<pg::Point>::ring_type&;
};
template <>
struct ring_mutable_type<pg::Polygon> {
  using type = model::polygon<pg::Point>::ring_type&;
};
template <>
struct interior_const_type<pg::Polygon> {
  using type = const model::polygon<pg::Point>::inner_container_type&;
};
template <>
struct interior_mutable_type<pg::Polygon> {
  using type = model::polygon<pg::Point>::inner_container_type&;
};

template <>
struct exterior_ring<pg::Polygon> {
  static auto get(model::polygon<pg::Point>& p)
      -> model::polygon<pg::Point>::ring_type& {
    return p.outer();
  }
  static auto get(model::polygon<pg::Point> const& p)
      -> model::polygon<pg::Point>::ring_type const& {
    return p.outer();
  }
};

template <>
struct interior_rings<pg::Polygon> {
  static auto get(model::polygon<pg::Point>& p)
      -> model::polygon<pg::Point>::inner_container_type& {
    return p.inners();
  }
  static auto get(model::polygon<pg::Point> const& p)
      -> model::polygon<pg::Point>::inner_container_type const& {
    return p.inners();
  }
};

}  // namespace boost::geometry::traits