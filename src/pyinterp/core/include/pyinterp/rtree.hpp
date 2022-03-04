// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <algorithm>
#include <functional>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/geodetic/coordinates.hpp"
#include "pyinterp/detail/geodetic/system.hpp"
#include "pyinterp/detail/geometry/rtree.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geodetic/system.hpp"

namespace pyinterp {

/// Type of radial functions exposed in the Python module.
using RadialBasisFunction = detail::math::RadialBasisFunction;

/// Type of window functions exposed in the Python module.
using WindowFunction = detail::math::window::Function;

/// RTree spatial index for geodetic point
///
/// @note
/// The tree of the "boost" library allows to directly handle the geodetic
/// coordinates, but it is much less efficient than the use of the tree in a
/// Cartesian space.
/// @tparam CoordinateType The class of storage for a point's coordinates.
/// @tparam Type The type of data stored in the tree.
/// @tparam N Number of dimensions in the Cartesian space handled.
template <typename CoordinateType, typename Type, size_t N>
class RTree : public detail::geometry::RTree<CoordinateType, Type, N> {
 public:
  /// The tree must at least store the ECEF coordinates
  static_assert(N >= 3,
                "The RTree must at least store the ECEF coordinates: x, y, z");

  /// Type of points handled by this instance
  using point_t =
      typename detail::geometry::RTree<CoordinateType, Type, N>::point_t;

  /// Type of distances between two points
  using distance_t =
      typename detail::geometry::RTree<CoordinateType, Type, N>::distance_t;

  /// Type of distances between two points.
  using result_t =
      typename detail::geometry::RTree<CoordinateType, Type, N>::result_t;

  /// Type of the implicit conversion between the type of coordinates and values
  using promotion_t =
      typename detail::geometry::RTree<CoordinateType, Type, N>::promotion_t;

  /// Pointer on the method converting LLA coordinates to ECEF.
  using Converter = point_t (RTree<CoordinateType, Type, N>::*)(
      const Eigen::Map<const Vector<CoordinateType>> &) const;

  /// Pointer on the method to Search for the nearest K nearest neighbors
  using Requester = std::vector<result_t> (RTree<CoordinateType, Type, N>::*)(
      const point_t &, const uint32_t) const;

  /// Default constructor
  explicit RTree(const std::optional<detail::geodetic::System> &wgs)
      : detail::geometry::RTree<CoordinateType, Type, N>(),
        coordinates_(wgs.value_or(detail::geodetic::System())) {}

  /// Returns the box able to contain all values stored in the container.
  ///
  /// @returns A tuple that contains the coordinates of the minimum and
  /// maximum corners of the box able to contain all values stored in the
  /// container or an empty tuple if there are no values in the container.
  [[nodiscard]] auto equatorial_bounds() const -> pybind11::tuple {
    if (this->empty()) {
      return pybind11::make_tuple();
    }

    auto x0 = std::numeric_limits<CoordinateType>::max();
    auto x1 = std::numeric_limits<CoordinateType>::min();
    auto y0 = std::numeric_limits<CoordinateType>::max();
    auto y1 = std::numeric_limits<CoordinateType>::min();
    auto z0 = std::numeric_limits<CoordinateType>::max();
    auto z1 = std::numeric_limits<CoordinateType>::min();

    std::for_each(this->tree_->begin(), this->tree_->end(),
                  [&](const auto &item) {
                    auto lla = to_lla(item.first);
                    x0 = std::min(x0, boost::geometry::get<0>(lla));
                    x1 = std::max(x1, boost::geometry::get<0>(lla));
                    y0 = std::min(y0, boost::geometry::get<1>(lla));
                    y1 = std::max(y1, boost::geometry::get<1>(lla));
                    z0 = std::min(z0, boost::geometry::get<2>(lla));
                    z1 = std::max(z1, boost::geometry::get<2>(lla));
                  });

    return pybind11::make_tuple(pybind11::make_tuple(x0, y0, z0),
                                pybind11::make_tuple(x1, y1, z1));
  }

  /// Populates the RTree with coordinates using the packaging algorithm
  ///
  /// @param coordinates Coordinates to be copied
  void packing(const pybind11::array_t<CoordinateType, pybind11::array::c_style>
                   &coordinates,
               const pybind11::array_t<Type> &values) {
    detail::check_array_ndim("coordinates", 2, coordinates);
    detail::check_array_ndim("values", 1, values);
    if (coordinates.shape(0) != values.size()) {
      throw std::invalid_argument(
          "coordinates, values could not be broadcast together with shape " +
          detail::ndarray_shape(coordinates) + ", " +
          detail::ndarray_shape(values));
    }
    switch (coordinates.shape(1)) {
      case N - 1:
        _packing<N - 1>(&RTree<CoordinateType, Type, N>::from_lon_lat,
                        coordinates, values);
        break;
      case N:
        _packing<N>(&RTree<CoordinateType, Type, N>::from_lon_lat_alt,
                    coordinates, values);
        break;
      default:
        throw std::invalid_argument(
            RTree<CoordinateType, Type, N>::invalid_shape());
    }
  }

  /// Insert new data into the search tree
  ///
  /// @param coordinates Coordinates to be copied
  void insert(const pybind11::array_t<CoordinateType, pybind11::array::c_style>
                  &coordinates,
              const pybind11::array_t<CoordinateType> &values) {
    detail::check_array_ndim("coordinates", 2, coordinates);
    detail::check_array_ndim("values", 1, values);
    if (coordinates.shape(0) != values.size()) {
      throw std::invalid_argument(
          "coordinates, values could not be broadcast together with shape " +
          detail::ndarray_shape(coordinates) + ", " +
          detail::ndarray_shape(values));
    }
    switch (coordinates.shape(1)) {
      case N - 1:
        _insert<N - 1>(&RTree<CoordinateType, Type, N>::from_lon_lat,
                       coordinates, values);
        break;
      case N:
        _insert<N>(&RTree<CoordinateType, Type, N>::from_lon_lat_alt,
                   coordinates, values);
        break;
      default:
        throw std::invalid_argument(
            RTree<CoordinateType, Type, N>::invalid_shape());
    }
  }

  /// Search for the nearest K nearest neighbors of a given coordinates.
  auto query(const pybind11::array_t<CoordinateType, pybind11::array::c_style>
                 &coordinates,
             const uint32_t k, const bool within,
             const size_t num_threads) const -> pybind11::tuple {
    detail::check_array_ndim("coordinates", 2, coordinates);
    switch (coordinates.shape(1)) {
      case N - 1:
        return _query<N - 1>(&RTree<CoordinateType, Type, N>::from_lon_lat,
                             coordinates, k, within, num_threads);
      case N:
        return _query<N>(&RTree<CoordinateType, Type, N>::from_lon_lat,
                         coordinates, k, within, num_threads);
      default:
        throw std::invalid_argument(
            RTree<CoordinateType, Type, N>::invalid_shape());
    }
  }

  /// TODO
  auto inverse_distance_weighting(
      const pybind11::array_t<CoordinateType, pybind11::array::c_style>
          &coordinates,
      const std::optional<distance_t> &radius, const uint32_t k,
      const uint32_t p, const bool within, const size_t num_threads) const
      -> pybind11::tuple {
    detail::check_array_ndim("coordinates", 2, coordinates);

    switch (coordinates.shape(1)) {
      case N - 1:
        return _inverse_distance_weighting<N - 1>(
            &RTree<CoordinateType, Type, N>::from_lon_lat, coordinates,
            radius.value_or(std::numeric_limits<distance_t>::max()), k, p,
            within, num_threads);
      case N:
        return _inverse_distance_weighting<N>(
            &RTree<CoordinateType, Type, N>::from_lon_lat, coordinates,
            radius.value_or(std::numeric_limits<distance_t>::max()), k, p,
            within, num_threads);
      default:
        throw std::invalid_argument(
            RTree<CoordinateType, Type, N>::invalid_shape());
    }
  }

  /// TODO
  auto radial_basis_function(
      const pybind11::array_t<CoordinateType, pybind11::array::c_style>
          &coordinates,
      const std::optional<distance_t> &radius, const uint32_t k,
      const RadialBasisFunction rbf, const std::optional<promotion_t> &epsilon,
      const promotion_t smooth, const bool within,
      const size_t num_threads) const -> pybind11::tuple {
    detail::check_array_ndim("coordinates", 2, coordinates);
    switch (coordinates.shape(1)) {
      case N - 1:
        return _rbf<N - 1>(
            &RTree<CoordinateType, Type, N>::from_lon_lat, coordinates,
            radius.value_or(std::numeric_limits<distance_t>::max()), k, rbf,
            epsilon.value_or(std::numeric_limits<promotion_t>::quiet_NaN()),
            smooth, within, num_threads);
      case N:
        return _rbf<N>(
            &RTree<CoordinateType, Type, N>::from_lon_lat_alt, coordinates,
            radius.value_or(std::numeric_limits<distance_t>::max()), k, rbf,
            epsilon.value_or(std::numeric_limits<promotion_t>::quiet_NaN()),
            smooth, within, num_threads);
      default:
        throw std::invalid_argument(
            RTree<CoordinateType, Type, N>::invalid_shape());
    }
  }

  /// TODO
  auto window_function(
      const pybind11::array_t<CoordinateType, pybind11::array::c_style>
          &coordinates,
      const distance_t &radius, const uint32_t k, const WindowFunction wf,
      const std::optional<distance_t> &arg, const bool within,
      const size_t num_threads) const -> pybind11::tuple {
    detail::check_array_ndim("coordinates", 2, coordinates);
    switch (coordinates.shape(1)) {
      case N - 1:
        return _window_function<N - 1>(
            &RTree<CoordinateType, Type, N>::from_lon_lat, coordinates, radius,
            k, wf, arg.value_or(0), within, num_threads);
      case N:
        return _window_function<N>(
            &RTree<CoordinateType, Type, N>::from_lon_lat_alt, coordinates,
            radius, k, wf, arg.value_or(0), within, num_threads);
      default:
        throw std::invalid_argument(
            RTree<CoordinateType, Type, N>::invalid_shape());
    }
  }

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    auto x = pybind11::array_t<CoordinateType>(
        pybind11::array::ShapeContainer{{this->size(), N}});
    auto u = pybind11::array_t<Type>(
        pybind11::array::ShapeContainer{{this->size()}});
    auto _x = x.template mutable_unchecked<2>();
    auto _u = u.template mutable_unchecked<1>();
    size_t ix = 0;
    std::for_each(this->tree_->begin(), this->tree_->end(),
                  [&](const auto &item) {
                    for (auto jx = 0UL; jx < N; ++jx) {
                      _x(ix, jx) = detail::geometry::point::get(item.first, jx);
                    }
                    _u(ix) = item.second;
                    ++ix;
                  });
    auto system = geodetic::System(this->coordinates_.system());
    return pybind11::make_tuple(system.getstate(), x, u);
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple &state)
      -> RTree<CoordinateType, Type, N> {
    if (state.size() != 3) {
      throw std::runtime_error("invalid state");
    }
    auto system = geodetic::System::setstate(state[0].cast<pybind11::tuple>());
    auto x = state[1].cast<pybind11::array_t<CoordinateType>>();
    auto u = state[2].cast<pybind11::array_t<Type>>();

    if (x.shape(1) != N || x.shape(0) != u.size()) {
      throw std::runtime_error("invalid state");
    }

    auto _x = x.template mutable_unchecked<2>();
    auto _u = u.template mutable_unchecked<1>();

    auto vector =
        std::vector<typename RTree<CoordinateType, Type, N>::value_t>();
    vector.reserve(u.size());

    auto point = point_t();

    for (auto ix = 0; ix < u.size(); ++ix) {
      for (auto jx = 0UL; jx < N; ++jx) {
        detail::geometry::point::set(point, _x(ix, jx), jx);
      }
      vector.emplace_back(std::make_pair(point, _u(ix)));
    }
    auto result = RTree<CoordinateType, Type, N>(system);
    static_cast<detail::geometry::RTree<CoordinateType, Type, N>>(result)
        .packing(vector);
    return result;
  }

 private:
  /// System for converting Geodetic coordinates into Cartesian coordinates.
  detail::geodetic::Coordinates coordinates_;

  /// Create the cartesian point for the given coordinates: longitude and
  /// latitude in degrees, altitude in meters, then the other coordinates
  /// defined in a Euclidean space.
  auto from_lon_lat_alt(
      const Eigen::Map<const Vector<CoordinateType>> &coordinates) const
      -> point_t {
    auto ecef = coordinates_.lla_to_ecef(
        detail::geometry::EquatorialPoint3D<CoordinateType>{
            coordinates(0), coordinates(1), coordinates(2)});
    auto result = point_t();

    boost::geometry::set<0>(result, boost::geometry::get<0>(ecef));
    boost::geometry::set<1>(result, boost::geometry::get<1>(ecef));
    boost::geometry::set<2>(result, boost::geometry::get<2>(ecef));

    for (auto ix = 3UL; ix < N; ++ix) {
      detail::geometry::point::set(result, coordinates(ix), ix);
    }
    return result;
  }

  /// Create the cartesian point for the given coordinates: longitude and
  /// latitude in degrees, then the other coordinated defined in a Euclidean
  /// space.
  auto from_lon_lat(const Eigen::Map<const Vector<CoordinateType>> &coordinates)
      const -> point_t {
    auto ecef = coordinates_.lla_to_ecef(
        detail::geometry::EquatorialPoint3D<CoordinateType>{coordinates(0),
                                                            coordinates(1), 0});
    auto result = point_t();

    boost::geometry::set<0>(result, boost::geometry::get<0>(ecef));
    boost::geometry::set<1>(result, boost::geometry::get<1>(ecef));
    boost::geometry::set<2>(result, boost::geometry::get<2>(ecef));

    for (auto ix = 2UL; ix < N - 1; ++ix) {
      detail::geometry::point::set(result, coordinates(ix), ix + 1);
    }
    return result;
  }

  /// Create the geographic point (latitude, longitude, and altitude) from the
  /// cartesian coordinates
  auto to_lla(const point_t &point) const
      -> detail::geometry::EquatorialPoint3D<CoordinateType> {
    return coordinates_.ecef_to_lla(detail::geometry::Point3D<CoordinateType>(
        boost::geometry::get<0>(point), boost::geometry::get<1>(point),
        boost::geometry::get<2>(point)));
  }

  /// Raise an exception if the size of the matrix does not conform to the
  /// number of coordinates handled by this instance.
  static auto invalid_shape() -> std::string {
    return "coordinates must be a matrix (n, " + std::to_string(N - 1) +
           ") to handle points defined by their longitudes, latitudes and "
           "other "
           "coordinates or a matrix (n," +
           std::to_string(N) +
           ") to handle points defined by their longitudes, latitudes, "
           "altitudes and other coordinates";
  }

  /// Packing coordinates
  ///
  /// @param coordinates Coordinates to be copied
  template <size_t M>
  void _packing(Converter converter,
                const pybind11::array_t<CoordinateType> &coordinates,
                const pybind11::array_t<Type> &values) {
    auto _coordinates = coordinates.template unchecked<2>();
    auto _values = values.template unchecked<1>();
    auto observations = coordinates.shape(0);
    auto vector =
        std::vector<typename RTree<CoordinateType, Type, N>::value_t>();

    vector.reserve(observations);

    for (auto ix = 0; ix < observations; ++ix) {
      vector.emplace_back(
          std::make_pair(std::invoke(converter, *this,
                                     Eigen::Map<const Vector<CoordinateType>>(
                                         &_coordinates(ix, 0), M)),
                         _values(ix)));
    }
    detail::geometry::RTree<CoordinateType, Type, N>::packing(vector);
  }

  /// Insert coordinates
  ///
  /// @param coordinates Coordinates to be copied
  template <size_t M>
  void _insert(Converter converter,
               const pybind11::array_t<CoordinateType> &coordinates,
               const pybind11::array_t<Type> &values) {
    auto _coordinates = coordinates.template unchecked<2>();
    auto _values = values.template unchecked<1>();

    for (auto ix = 0; ix < coordinates.shape(0); ++ix) {
      detail::geometry::RTree<CoordinateType, Type, N>::insert(
          std::make_pair(std::invoke(converter, *this,
                                     Eigen::Map<const Vector<CoordinateType>>(
                                         &_coordinates(ix, 0), M)),
                         _values(ix)));
    }
  }

  /// Search for the nearest K nearest neighbors of a given coordinates.
  template <size_t M>
  auto _query(Converter converter,
              const pybind11::array_t<CoordinateType> &coordinates,
              const uint32_t k, const bool within,
              const size_t num_threads) const -> pybind11::tuple {
    Requester requester =
        within ? &detail::geometry::RTree<CoordinateType, Type, N>::query_within
               : &detail::geometry::RTree<CoordinateType, Type, N>::query;

    auto _coordinates = coordinates.template unchecked<2>();
    auto size = coordinates.shape(0);

    // Allocation of result matrices.
    auto distance =
        pybind11::array_t<distance_t>(pybind11::array::ShapeContainer{
            size, static_cast<pybind11::ssize_t>(k)});
    auto value = pybind11::array_t<Type>(pybind11::array::ShapeContainer{
        size, static_cast<pybind11::ssize_t>(k)});

    auto _distance = distance.template mutable_unchecked<2>();
    auto _value = value.template mutable_unchecked<2>();

    {
      pybind11::gil_scoped_release release;

      // Captures the detected exceptions in the calculation function
      // (only the last exception captured is kept)
      auto except = std::exception_ptr(nullptr);

      detail::dispatch(
          [&](size_t start, size_t end) {
            try {
              auto point = point_t();

              for (size_t ix = start; ix < end; ++ix) {
                point = std::move(
                    std::invoke(converter, *this,
                                Eigen::Map<const Vector<CoordinateType>>(
                                    &_coordinates(ix, 0), M)));

                auto nearest = std::invoke(requester, *this, point, k);
                auto jx = 0ULL;

                // Fill in the calculation result for all neighbors found
                for (; jx < nearest.size(); ++jx) {
                  _distance(ix, jx) = nearest[jx].first;
                  _value(ix, jx) = nearest[jx].second;
                }

                // The rest of the result is filled with invalid values.
                for (; jx < k; ++jx) {
                  _distance(ix, jx) = -1;
                  _value(ix, jx) = Type(-1);
                }
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
    return pybind11::make_tuple(distance, value);
  }

  /// Inverse distance weighting interpolation
  template <size_t M>
  auto _inverse_distance_weighting(
      Converter converter, const pybind11::array_t<CoordinateType> &coordinates,
      const distance_t radius, const uint32_t k, const uint32_t p,
      const bool within, const size_t num_threads) const -> pybind11::tuple {
    auto _coordinates = coordinates.template unchecked<2>();
    auto size = coordinates.shape(0);

    // Allocation of result vectors.
    auto data =
        pybind11::array_t<distance_t>(pybind11::array::ShapeContainer{size});
    auto neighbors =
        pybind11::array_t<uint32_t>(pybind11::array::ShapeContainer{size});

    auto _data = data.template mutable_unchecked<1>();
    auto _neighbors = neighbors.template mutable_unchecked<1>();

    {
      pybind11::gil_scoped_release release;

      // Captures the detected exceptions in the calculation function
      // (only the last exception captured is kept)
      auto except = std::exception_ptr(nullptr);

      detail::dispatch(
          [&](size_t start, size_t end) {
            try {
              auto point = point_t();

              for (size_t ix = start; ix < end; ++ix) {
                point = std::move(
                    std::invoke(converter, *this,
                                Eigen::Map<const Vector<CoordinateType>>(
                                    &_coordinates(ix, 0), M)));

                auto result = detail::geometry::RTree<CoordinateType, Type, N>::
                    inverse_distance_weighting(point, radius, k, p, within);
                _data(ix) = result.first;
                _neighbors(ix) = result.second;
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
    return pybind11::make_tuple(data, neighbors);
  }

  /// Radial basis function interpolation
  template <size_t M>
  auto _rbf(Converter converter,
            const pybind11::array_t<CoordinateType> &coordinates,
            const distance_t radius, const uint32_t k,
            const RadialBasisFunction rbf, const promotion_t epsilon,
            const promotion_t smooth, const bool within,
            const size_t num_threads) const -> pybind11::tuple {
    auto _coordinates = coordinates.template unchecked<2>();
    auto size = coordinates.shape(0);

    // Construction of the interpolator.
    auto rbf_handler = detail::math::RBF<promotion_t>(epsilon, smooth, rbf);

    // Allocation of result vectors.
    auto data =
        pybind11::array_t<promotion_t>(pybind11::array::ShapeContainer{size});
    auto neighbors =
        pybind11::array_t<uint32_t>(pybind11::array::ShapeContainer{size});

    auto _data = data.template mutable_unchecked<1>();
    auto _neighbors = neighbors.template mutable_unchecked<1>();

    {
      pybind11::gil_scoped_release release;

      // Captures the detected exceptions in the calculation function
      // (only the last exception captured is kept)
      auto except = std::exception_ptr(nullptr);

      detail::dispatch(
          [&](size_t start, size_t end) {
            try {
              auto point = point_t();

              for (size_t ix = start; ix < end; ++ix) {
                point = std::move(
                    std::invoke(converter, *this,
                                Eigen::Map<const Vector<CoordinateType>>(
                                    &_coordinates(ix, 0), M)));

                auto result = detail::geometry::RTree<
                    CoordinateType, Type, N>::radial_basis_function(point,
                                                                    rbf_handler,
                                                                    radius, k,
                                                                    within);
                _data(ix) = result.first;
                _neighbors(ix) = result.second;
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
    return pybind11::make_tuple(data, neighbors);
  }

  /// Window function interpolation
  template <size_t M>
  auto _window_function(Converter converter,
                        const pybind11::array_t<CoordinateType> &coordinates,
                        const distance_t radius, const uint32_t k,
                        const WindowFunction wf, const distance_t arg,
                        const bool within, const size_t num_threads) const
      -> pybind11::tuple {
    auto _coordinates = coordinates.template unchecked<2>();
    auto size = coordinates.shape(0);

    // Allocation of result vectors.
    auto data =
        pybind11::array_t<distance_t>(pybind11::array::ShapeContainer{size});
    auto neighbors =
        pybind11::array_t<uint32_t>(pybind11::array::ShapeContainer{size});

    auto wf_handler = detail::math::WindowFunction<distance_t>(wf);

    auto _data = data.template mutable_unchecked<1>();
    auto _neighbors = neighbors.template mutable_unchecked<1>();

    {
      pybind11::gil_scoped_release release;

      // Captures the detected exceptions in the calculation function
      // (only the last exception captured is kept)
      auto except = std::exception_ptr(nullptr);

      detail::dispatch(
          [&](size_t start, size_t end) {
            try {
              auto point = point_t();

              for (size_t ix = start; ix < end; ++ix) {
                point = std::move(
                    std::invoke(converter, *this,
                                Eigen::Map<const Vector<CoordinateType>>(
                                    &_coordinates(ix, 0), M)));

                auto result =
                    detail::geometry::RTree<CoordinateType, Type,
                                            N>::window_function(point,
                                                                wf_handler, arg,
                                                                radius, k,
                                                                within);
                _data(ix) = result.first;
                _neighbors(ix) = result.second;
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
    return pybind11::make_tuple(data, neighbors);
  }
};

}  // namespace pyinterp
