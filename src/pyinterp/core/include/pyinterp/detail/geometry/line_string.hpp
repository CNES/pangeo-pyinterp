// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <boost/geometry.hpp>
#include <sstream>
#include <utility>

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/geometry/point.hpp"

namespace pyinterp::detail::geometry {

/// A linestring (named so by OGC) is a collection of points.
template <typename T>
class LineString : public boost::geometry::model::linestring<Point2D<T>> {
 public:
  using Base = boost::geometry::model::linestring<Point2D<T>>;

  /// Default constructor
  LineString() = default;

  /// Build a new line string with the coordinates provided.
  ///
  /// @param x The x coordinates of the points.
  /// @param y The y coordinates of the points.
  LineString(const Eigen::Ref<const Vector<T>>& x,
             const Eigen::Ref<const Vector<T>>& y) {
    check_eigen_shape("x", x, "y", y);
    for (auto ix = static_cast<Eigen::Index>(0); ix < x.size(); ++ix) {
      Base::emplace_back(Point2D<T>{x(ix), y(ix)});
    }
  }

  /// Test if this linestring intersects with another linestring.
  ///
  /// @param rhs the linestring to test.
  /// @return true if the linestrings intersect.
  [[nodiscard]] auto intersects(const LineString& rhs) const -> bool {
    return boost::geometry::intersects(*this, rhs);
  }

  /// Compute the intersection of this linestring with another linestring.
  ///
  /// @param rhs the linestring to test.
  /// @return the intersection of the two linestrings.
  [[nodiscard]] auto intersection(const LineString& rhs) const -> LineString {
    LineString output;
    boost::geometry::intersection(*this, rhs, output);
    return output;
  }

  /// Returns a read-only (constant) iterator that points to the
  /// first point of the linestring.
  [[nodiscard]] auto begin() const -> decltype(Base::begin()) {
    return Base::begin();
  }

  /// Returns an iterator that points to the first point of the linestring.
  [[nodiscard]] auto begin() -> decltype(Base::begin()) {
    return Base::begin();
  }

  /// Returns a read-only (constant) iterator that points to the
  /// last point of the linestring.
  [[nodiscard]] auto end() const -> decltype(Base::end()) {
    return Base::end();
  }

  /// Returns an iterator that points to the last point of the linestring.
  [[nodiscard]] auto end() -> decltype(Base::end()) { return Base::end(); }
};

}  // namespace pyinterp::detail::geometry

namespace boost::geometry::traits {
namespace pdg = pyinterp::detail::geometry;

template <typename T>
struct tag<pdg::LineString<T>> {
  using type = linestring_tag;
};

}  // namespace boost::geometry::traits
