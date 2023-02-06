#include "pyinterp/delaunay.hpp"

#include <CGAL/surface_neighbors_3.h>

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/thread.hpp"

namespace pyinterp {

Delaunay::Delaunay(const NDArray& coordinates,
                   const pybind11::array_t<double>& values,
                   const std::optional<detail::geodetic::Spheroid>& spheroid)
    : triangulation_(),
      point_values_(),
      coordinates_(spheroid.value_or(detail::geodetic::Spheroid())) {
  detail::check_array_ndim("coordinates", 2, coordinates);
  detail::check_array_ndim("values", 1, values);
  if (coordinates.shape(0) != values.size()) {
    throw std::invalid_argument(
        "coordinates, values could not be broadcast together with shape " +
        detail::ndarray_shape(coordinates) + ", " +
        detail::ndarray_shape(values));
  }
  if (coordinates.shape(1) != 2 && coordinates.shape(1) != 3) {
    throw std::invalid_argument(
        "coordinates must be a 2D array with 2 or 3 columns");
  }
  auto points = std::vector<KPoint>();
  points.reserve(coordinates.shape(0));

  auto coordinates_view = coordinates.unchecked<2>();
  auto values_view = values.unchecked<1>();

  auto have_altitude = coordinates.shape(1) == 3;

  {
    auto gil = pybind11::gil_scoped_release();

    for (auto i = 0; i < coordinates.shape(0); ++i) {
      auto x = coordinates_view(i, 0);
      auto y = coordinates_view(i, 1);
      auto z = have_altitude ? coordinates_view(i, 2) : 0.0;
      auto ecef = coordinates_.lla_to_ecef(
          detail::geometry::EquatorialPoint3D<double>{x, y, z});
      auto point = KPoint(ecef.get<0>(), ecef.get<1>(), ecef.get<2>());
      points.emplace_back(point);
      point_values_.emplace(point, values_view(i));
    }
    triangulation_.insert(points.begin(), points.end());
  }
}

auto Delaunay::natural_neighbor_interpolation(const NDArray& coordinates,
                                              const size_t num_threads) const
    -> pybind11::array_t<double> {
  detail::check_array_ndim("coordinates", 2, coordinates);
  if (coordinates.shape(1) != 2 && coordinates.shape(1) != 3) {
    throw std::invalid_argument(
        "coordinates must be a 2D array with 2 or 3 columns");
  }
  auto have_altitude = coordinates.shape(1) == 3;
  auto result = pybind11::array_t<double>(coordinates.shape(0));
  auto result_view = result.mutable_unchecked<1>();
  auto coordinates_view = coordinates.unchecked<2>();
  auto except = std::exception_ptr(nullptr);

  auto worker = [&](size_t begin, size_t end) {
    try {
      auto fh = DelaunayTriangulation::Cell_handle();

      for (auto ix = begin; ix < end; ++ix) {
        auto x = coordinates_view(ix, 0);
        auto y = coordinates_view(ix, 1);
        auto z = have_altitude ? coordinates_view(ix, 2) : 0.0;
        auto ecef = coordinates_.lla_to_ecef(
            detail::geometry::EquatorialPoint3D<double>{x, y, z});
        auto point = KPoint(ecef.get<0>(), ecef.get<1>(), ecef.get<2>());
        fh = triangulation_.locate(point, fh);

        auto normal = Kernel::Vector_3(point - CGAL::ORIGIN);
        auto coordinates_3 = std::vector<std::pair<KPoint, Kernel::FT>>();
        auto snc3 = CGAL::surface_neighbor_coordinates_3(
            triangulation_, point, normal, std::back_inserter(coordinates_3),
            fh);
        if (!snc3.third) {
          result_view(ix) = std::numeric_limits<double>::quiet_NaN();
        } else {
          result_view(ix) = CGAL::linear_interpolation(
              coordinates_3.begin(), coordinates_3.end(), snc3.second,
              CGAL::Data_access<PointValueMap>(point_values_));
        }
      }
    } catch (...) {
      except = std::current_exception();
    }
  };

  {
    auto gil = pybind11::gil_scoped_release();

    detail::dispatch(worker, coordinates.shape(0), num_threads);
    if (except != nullptr) {
      std::rethrow_exception(except);
    }
  }
  return result;
}

}  // namespace pyinterp
