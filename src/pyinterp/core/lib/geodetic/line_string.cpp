// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geodetic/line_string.hpp"

#include <boost/geometry/algorithms/simplify.hpp>

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/geodetic/multipolygon.hpp"
#include "pyinterp/geodetic/polygon.hpp"

namespace pyinterp::geodetic {

LineString::LineString(const Eigen::Ref<const Eigen::VectorXd>& lon,
                       const Eigen::Ref<const Eigen::VectorXd>& lat) {
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

auto LineString::intersection(const Polygon& rhs,
                              const std::optional<Spheroid>& wgs) const
    -> std::vector<LineString> {
  std::vector<LineString> output;
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

  auto x = Eigen::Map<const Eigen::VectorXd>(lon.data(), lon.size());
  auto y = Eigen::Map<const Eigen::VectorXd>(lat.data(), lat.size());
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
                               Eigen::VectorXd& result) {
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
    -> Eigen::VectorXd {
  if (size() == 0) {
    return Eigen::VectorXd{};
  }

  auto spheroid =
      wgs.has_value()
          ? static_cast<boost::geometry::srs::spheroid<double>>(*wgs)
          : boost::geometry::srs::spheroid<double>();
  auto result = Eigen::VectorXd(size());

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

auto LineString::simplify(const double tolerance,
                          const DistanceStrategy strategy,
                          const std::optional<Spheroid>& wgs) const
    -> LineString {
  auto spheroid =
      wgs.has_value()
          ? static_cast<boost::geometry::srs::spheroid<double>>(*wgs)
          : boost::geometry::srs::spheroid<double>();
  auto result = LineString{};

  using SimplifyAndoyer = boost::geometry::strategies::simplify::geographic<
      boost::geometry::strategy::andoyer>;
  using SimplifyThomas = boost::geometry::strategies::simplify::geographic<
      boost::geometry::strategy::thomas>;
  using SimplifyVincenty = boost::geometry::strategies::simplify::geographic<
      boost::geometry::strategy::vincenty>;

  switch (strategy) {
    case kAndoyer:
      boost::geometry::simplify(*this, result, tolerance,
                                SimplifyAndoyer(spheroid));
      break;
    case kThomas:
      boost::geometry::simplify(*this, result, tolerance,
                                SimplifyThomas(spheroid));
      break;
    case kVincenty:
      boost::geometry::simplify(*this, result, tolerance,
                                SimplifyVincenty(spheroid));
      break;
    default:
      throw std::invalid_argument("unknown strategy: " +
                                  std::to_string(static_cast<int>(strategy)));
  }
  return result;
}

namespace impl {

inline auto closest_point(const LineString& ls, const Point& point) -> Point {
  auto segment = boost::geometry::model::segment<Point>{};
  boost::geometry::closest_points(ls, point, segment);
  return segment.first;
}

inline auto closest_point(const LineString& ls, const Point& point,
                          const Spheroid& wgs) -> Point {
  auto segment = boost::geometry::model::segment<Point>{};
  auto strategy = boost::geometry::strategies::closest_points::geographic<>(
      static_cast<boost::geometry::srs::spheroid<double>>(wgs));
  boost::geometry::closest_points(ls, point, segment, strategy);
  return segment.first;
}

}  // namespace impl

auto LineString::closest_point(const Point& point,
                               const std::optional<Spheroid>& wgs) const
    -> Point {
  return wgs.has_value() ? impl::closest_point(*this, point, *wgs)
                         : impl::closest_point(*this, point);
}

auto LineString::closest_point(const Eigen::Ref<const Eigen::VectorXd>& lon,
                               const Eigen::Ref<const Eigen::VectorXd>& lat,
                               const std::optional<Spheroid>& wgs,
                               const size_t num_threads) const
    -> std::tuple<Eigen::VectorXd, Eigen::VectorXd> {
  auto result_lon = Eigen::VectorXd(lon.size());
  auto result_lat = Eigen::VectorXd(lat.size());
  auto except = std::exception_ptr(nullptr);

  if (wgs.has_value()) {
    auto worker = [&](size_t start, size_t end) {
      try {
        for (auto ix = start; ix < end; ++ix) {
          auto point = impl::closest_point(*this, {lon[ix], lat[ix]}, *wgs);
          result_lon[ix] = point.lon();
          result_lat[ix] = point.lat();
        }
      } catch (...) {
        except = std::current_exception();
      }
    };
    detail::dispatch(worker, lon.size(), num_threads);
  } else {
    auto worker = [&](size_t start, size_t end) {
      try {
        for (auto ix = start; ix < end; ++ix) {
          auto point = impl::closest_point(*this, {lon[ix], lat[ix]});
          result_lon[ix] = point.lon();
          result_lat[ix] = point.lat();
        }
      } catch (...) {
        except = std::current_exception();
      }
    };
    detail::dispatch(worker, lon.size(), num_threads);
  }

  if (except != nullptr) {
    std::rethrow_exception(except);
  }
  return std::make_tuple(result_lon, result_lat);
}

}  // namespace pyinterp::geodetic
