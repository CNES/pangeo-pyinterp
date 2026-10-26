// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

/// @brief Cheap rejection of the points outside a container.
///
/// Testing a point against a polygon walks all its edges; testing it against
/// the envelope of the polygon first rejects most of the points outside in
/// constant time. The envelope is closed, so it never rejects a point that is
/// within or covered by the container.
///
/// On the ellipsoid, the envelope of a polygon is computed from its boundary:
/// it does not enclose a pole the polygon contains. The filter is disabled in
/// that case.
///
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @tparam Point Point type of the tested points
/// @tparam Container Container type
template <GeometryNamespace NS, typename Point, typename Container>
class EnvelopeFilter {
 public:
  /// @param[in] container The container the points are tested against
  explicit EnvelopeFilter(const Container& container)
      : envelope_{boost::geometry::return_envelope<Box>(container)} {
    if constexpr (NS == GeometryNamespace::kGeographic) {
      enabled_ = !boost::geometry::covered_by(Point{0.0, 90.0}, container) &&
                 !boost::geometry::covered_by(Point{0.0, -90.0}, container);
    }
  }

  /// @brief False if the point is certainly outside the container
  [[nodiscard]] auto may_contain(const Point& point) const -> bool {
    return !enabled_ || boost::geometry::covered_by(point, envelope_);
  }

 private:
  using Box = boost::geometry::model::box<Point>;

  Box envelope_;
  bool enabled_{true};
};

}  // namespace pyinterp::geometry::pybind
