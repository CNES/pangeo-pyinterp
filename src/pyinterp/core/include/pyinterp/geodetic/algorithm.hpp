#pragma once
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/covered_by.hpp>
#include <boost/geometry/srs/spheroid.hpp>
#if BOOST_VERSION >= 107500
#include <boost/geometry/strategy/area.hpp>
#include <boost/geometry/strategy/geographic/area.hpp>
#else
#include <boost/geometry/strategies/area.hpp>
#include <boost/geometry/strategies/geographic/area.hpp>
#endif
#include <optional>

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/geodetic/system.hpp"

namespace pyinterp::geodetic {

/// Calculate the area
template <typename Geometry>
[[nodiscard]] inline auto area(const Geometry &geometry,
                               const std::optional<System> &wgs) -> double {
  auto spheroid = wgs.has_value()
                      ? boost::geometry::srs::spheroid(wgs->semi_major_axis(),
                                                       wgs->semi_minor_axis())
                      : boost::geometry::srs::spheroid<double>();
  auto strategy = boost::geometry::strategy::area::geographic<
      boost::geometry::strategy::vincenty, 5>(spheroid);
  return boost::geometry::area(geometry, strategy);
}

/// Checks if the first geometry is inside or on border the second geometry
/// using the specified strategy.
template <typename Geometry1, typename Geometry2>
[[nodiscard]] inline auto covered_by(
    const Geometry2 &geometry2, const Eigen::Ref<const Eigen::VectorXd> &lon,
    const Eigen::Ref<const Eigen::VectorXd> &lat, const size_t num_threads)
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
              _result(ix) = static_cast<int8_t>(boost::geometry::covered_by(
                  Geometry1(lon(ix), lat(ix)), geometry2));
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

}  // namespace pyinterp::geodetic