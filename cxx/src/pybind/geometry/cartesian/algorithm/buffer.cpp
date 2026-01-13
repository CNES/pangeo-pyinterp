// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include <boost/geometry.hpp>
#include <boost/geometry/core/cs.hpp>

#include "pyinterp/geometry/cartesian/linestring.hpp"
#include "pyinterp/geometry/cartesian/multi_linestring.hpp"
#include "pyinterp/geometry/cartesian/multi_point.hpp"
#include "pyinterp/geometry/cartesian/multi_polygon.hpp"
#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/geometry/cartesian/polygon.hpp"
#include "pyinterp/geometry/cartesian/ring.hpp"

namespace nb = nanobind;
namespace bg = boost::geometry;
using nb::literals::operator""_a;

namespace pyinterp::geometry::cartesian::pybind {

// ============================================================================
// Distance Strategies
// ============================================================================

class DistanceSymmetric {
 public:
  explicit DistanceSymmetric(double buffer_distance)
      : buffer_distance_(buffer_distance) {}

  [[nodiscard]] auto get() const
      -> bg::strategy::buffer::distance_symmetric<double> {
    return bg::strategy::buffer::distance_symmetric<double>(buffer_distance_);
  }

 private:
  double buffer_distance_;
};

class DistanceAsymmetric {
 public:
  DistanceAsymmetric(double distance_left, double distance_right)
      : distance_left_(distance_left), distance_right_(distance_right) {}

  [[nodiscard]] auto get() const
      -> bg::strategy::buffer::distance_asymmetric<double> {
    return {distance_left_, distance_right_};
  }

 private:
  double distance_left_;
  double distance_right_;
};

// ============================================================================
// Join Strategies
// ============================================================================

class JoinRound {
 public:
  explicit JoinRound(int points_per_circle = 36)
      : points_per_circle_(points_per_circle) {}

  [[nodiscard]] auto get() const -> bg::strategy::buffer::join_round {
    return bg::strategy::buffer::join_round(points_per_circle_);
  }

 private:
  int points_per_circle_;
};

class JoinMiter {
 public:
  explicit JoinMiter(double miter_limit = 5.0) : miter_limit_(miter_limit) {}

  [[nodiscard]] auto get() const -> bg::strategy::buffer::join_miter {
    return bg::strategy::buffer::join_miter(miter_limit_);
  }

 private:
  double miter_limit_;
};

// ============================================================================
// End Strategies
// ============================================================================

class EndRound {
 public:
  explicit EndRound(int points_per_circle = 36)
      : points_per_circle_(points_per_circle) {}

  [[nodiscard]] auto get() const -> bg::strategy::buffer::end_round {
    return bg::strategy::buffer::end_round(points_per_circle_);
  }

 private:
  int points_per_circle_;
};

class EndFlat {
 public:
  [[nodiscard]] static auto get() -> bg::strategy::buffer::end_flat {
    return {};
  }
};

// ============================================================================
// Point Strategies
// ============================================================================

class PointCircle {
 public:
  explicit PointCircle(int points_per_circle = 36)
      : points_per_circle_(points_per_circle) {}

  [[nodiscard]] auto get() const -> bg::strategy::buffer::point_circle {
    return bg::strategy::buffer::point_circle(points_per_circle_);
  }

 private:
  int points_per_circle_;
};

class PointSquare {
 public:
  [[nodiscard]] static auto get() -> bg::strategy::buffer::point_square {
    return {};
  }
};

// ============================================================================
// Side Strategies
// ============================================================================

class SideStraight {
 public:
  [[nodiscard]] static auto get() -> bg::strategy::buffer::side_straight {
    return {};
  }
};

// ============================================================================
// Buffer Function
// ============================================================================

constexpr auto kBufferDoc = R"doc(
Calculate the buffer of a geometry.

The buffer algorithm creates a polygon representing all points within a
specified distance from the input geometry.

Args:
    geometry: Input geometry (Point, LineString, Ring, Polygon, MultiPoint,
        MultiLineString, or MultiPolygon).
    distance_strategy: Distance strategy object (DistanceSymmetric or
        DistanceAsymmetric). Controls the buffer distance.
    side_strategy: Side strategy object (SideStraight). Controls side
        generation.
    join_strategy: Join strategy object (JoinRound or JoinMiter).
        Controls corner generation.
    end_strategy: End strategy object (EndRound or EndFlat).
        Controls linestring end generation.
    point_strategy: Point strategy object (PointCircle or PointSquare).
        Controls point buffer shape.

Returns:
    MultiPolygon: The buffered geometry.
)doc";

// Type aliases for variant types
using DistanceStrategy = std::variant<DistanceSymmetric, DistanceAsymmetric>;
using JoinStrategy = std::variant<JoinRound, JoinMiter>;
using EndStrategy = std::variant<EndRound, EndFlat>;
using PointStrategy = std::variant<PointCircle, PointSquare>;

// Generic buffer implementation
template <typename Geometry>
auto buffer_impl(const Geometry& geometry, const DistanceStrategy& distance,
                 const JoinStrategy& join, const EndStrategy& end,
                 const PointStrategy& point) -> MultiPolygon {
  MultiPolygon result;
  {
    nb::gil_scoped_release release;

    std::visit(
        [&](const auto& dist, const auto& j, const auto& e,
            const auto& p) -> auto {
          bg::buffer(geometry, result, dist.get(), SideStraight::get(), j.get(),
                     e.get(), p.get());
        },
        distance, join, end, point);
  }
  return result;
}

// Bind buffer for a single geometry type
template <typename Geometry>
void bind_buffer_for_geometry(nb::module_& m) {
  m.def(
      "buffer",
      [](const Geometry& geometry, const DistanceStrategy& distance,
         const JoinStrategy& join, const EndStrategy& end,
         const PointStrategy& point) -> MultiPolygon {
        return buffer_impl(geometry, distance, join, end, point);
      },
      "geometry"_a, "distance_strategy"_a, "join_strategy"_a, "end_strategy"_a,
      "point_strategy"_a, kBufferDoc);
}

auto init_buffer(nb::module_& m) -> void {
  // Bind distance strategies
  nb::class_<DistanceSymmetric>(m, "DistanceSymmetric")
      .def(nb::init<double>(), "buffer_distance"_a,
           "Create symmetric distance strategy");

  nb::class_<DistanceAsymmetric>(m, "DistanceAsymmetric")
      .def(nb::init<double, double>(), "distance_left"_a, "distance_right"_a,
           "Create asymmetric distance strategy");

  // Bind join strategies
  nb::class_<JoinRound>(m, "JoinRound")
      .def(nb::init<int>(), "points_per_circle"_a = 36,
           "Create rounded join strategy");

  nb::class_<JoinMiter>(m, "JoinMiter")
      .def(nb::init<double>(), "miter_limit"_a = 5.0,
           "Create miter join strategy");

  // Bind end strategies
  nb::class_<EndRound>(m, "EndRound")
      .def(nb::init<int>(), "points_per_circle"_a = 36,
           "Create rounded end strategy");

  nb::class_<EndFlat>(m, "EndFlat")
      .def(nb::init<>(), "Create flat end strategy");

  // Bind point strategies
  nb::class_<PointCircle>(m, "PointCircle")
      .def(nb::init<int>(), "points_per_circle"_a = 36,
           "Create circular point strategy");

  nb::class_<PointSquare>(m, "PointSquare")
      .def(nb::init<>(), "Create square point strategy");

  // Bind side strategy
  nb::class_<SideStraight>(m, "SideStraight")
      .def(nb::init<>(), "Create straight side strategy");

  bind_buffer_for_geometry<Point>(m);
  bind_buffer_for_geometry<LineString>(m);
  bind_buffer_for_geometry<Ring>(m);
  bind_buffer_for_geometry<Polygon>(m);
  bind_buffer_for_geometry<MultiPoint>(m);
  bind_buffer_for_geometry<MultiLineString>(m);
  bind_buffer_for_geometry<MultiPolygon>(m);
}

}  // namespace pyinterp::geometry::cartesian::pybind
