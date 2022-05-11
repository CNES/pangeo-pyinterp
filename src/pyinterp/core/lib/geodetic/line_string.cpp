// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geodetic/line_string.hpp"

#include "pyinterp/detail/broadcast.hpp"

namespace pyinterp::geodetic {

LineString::LineString(const Eigen::Ref<const Vector<double>>& lon,
                       const Eigen::Ref<const Vector<double>>& lat) {
  detail::check_eigen_shape("lon", lon, "lat", lat);
  for (auto ix = static_cast<Eigen::Index>(0); ix < lon.size(); ++ix) {
    Base::emplace_back(Point{lon(ix), lat(ix)});
  }
}

auto LineString::from_geojson(const pybind11::list& array) -> LineString {
  auto result = LineString{};
  auto* base = dynamic_cast<Base*>(&result);
  for (auto point : array) {
    base->push_back(Point::from_geojson(point.cast<pybind11::list>()));
  }
  return result;
}

auto LineString::intersects(const LineString& rhs,
                            const std::optional<Spheroid>& wgs) const -> bool {
  if (wgs) {
    return boost::geometry::intersects(
        *this, rhs,
        boost::geometry::strategy::intersection::geographic_segments<>(
            static_cast<boost::geometry::srs::spheroid<double>>(*wgs)));
  } else {
    return boost::geometry::intersects(*this, rhs);
  }
}

auto LineString::intersection(const LineString& rhs,
                              const std::optional<Spheroid>& wgs) const
    -> LineString {
  LineString output;
  if (wgs) {
    boost::geometry::intersection(
        *this, rhs, output,
        boost::geometry::strategy::intersection::geographic_segments<>(
            static_cast<boost::geometry::srs::spheroid<double>>(*wgs)));
  } else {
    boost::geometry::intersection(*this, rhs, output);
  }
  return output;
}

auto LineString::getstate() const -> pybind11::tuple {
  auto lon = pybind11::array_t<double>(pybind11::array::ShapeContainer{size()});
  auto lat = pybind11::array_t<double>(pybind11::array::ShapeContainer{size()});
  auto _lon = lon.mutable_unchecked<1>();
  auto _lat = lat.mutable_unchecked<1>();
  auto ix = static_cast<int64_t>(0);
  for (const auto& item : *this) {
    _lon[ix] = item.lon();
    _lat[ix] = item.lat();
    ++ix;
  }
  return pybind11::make_tuple(lon, lat);
}

auto LineString::setstate(const pybind11::tuple& state) -> LineString {
  if (state.size() != 2) {
    throw std::runtime_error("invalid state");
  }
  auto lon = state[0].cast<pybind11::array_t<double>>();
  auto lat = state[1].cast<pybind11::array_t<double>>();

  auto x = Eigen::Map<const Vector<double>>(lon.data(), lon.size());
  auto y = Eigen::Map<const Vector<double>>(lat.data(), lat.size());
  return {x, y};
}

auto LineString::to_geojson() const -> pybind11::dict {
  auto result = pybind11::dict();
  result["type"] = "LineString";
  auto coordinates = pybind11::list();
  for (auto& point : *this) {
    coordinates.append(point.coordinates());
  }
  result["coordinates"] = coordinates;
  return result;
}

template <typename Strategy>
auto curvilinear_distance_impl(const LineString& ls, const Strategy& strategy,
                               Vector<double>& result) {
  auto total_distance = static_cast<double>(0);
  result[0] = total_distance;

  auto it = ls.begin() + 1;

  for (auto ix = static_cast<size_t>(1); ix < ls.size(); ++ix) {
    auto distance = boost::geometry::distance(*std::prev(it), *it, strategy);
    total_distance += distance;
    result[ix] = total_distance;
    ++it;
  }
}

auto LineString::curvilinear_distance(DistanceStrategy strategy,
                                      const std::optional<Spheroid>& wgs) const
    -> Vector<double> {
  if (size() == 0) {
    return Vector<double>{};
  }

  auto spheroid =
      wgs.has_value()
          ? static_cast<boost::geometry::srs::spheroid<double>>(*wgs)
          : boost::geometry::srs::spheroid<double>();
  auto result = Vector<double>(size());

  switch (strategy) {
    case kAndoyer:
      curvilinear_distance_impl(*this, Andoyer(spheroid), result);
      break;
    case kThomas:
      curvilinear_distance_impl(*this, Thomas(spheroid), result);
      break;
    case kVincenty:
      curvilinear_distance_impl(*this, Vincenty(spheroid), result);
      break;
    default:
      throw std::invalid_argument("unknown strategy: " +
                                  std::to_string(static_cast<int>(strategy)));
  }
  return result;
}

}  // namespace pyinterp::geodetic
