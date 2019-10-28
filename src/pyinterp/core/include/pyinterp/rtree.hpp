// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>
#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/geodetic/rtree.hpp"
#include "pyinterp/detail/geodetic/system.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/geodetic/system.hpp"

namespace pyinterp {

template <typename Coordinate, typename Type>
class RTree : public detail::geodetic::RTree<Coordinate, Type> {
 public:
  /// Type of distances between two points
  using distance_t =
      typename detail::geodetic::RTree<Coordinate, Type>::distance_t;

  /// Inherit constructors
  using detail::geodetic::RTree<Coordinate, Type>::RTree;

  /// Populates the RTree with coordinates using the packaging algorithm
  ///
  /// @param coordinates Coordinates to be copied
  void packing(const pybind11::array_t<Coordinate> &coordinates,
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
      case 2:
        _packing<2>(coordinates, values);
        break;
      case 3:
        _packing<3>(coordinates, values);
        break;
      default:
        throw std::invalid_argument(
            "coordinates must be a matrix (n, 2) to add points defined by "
            "their longitudes and latitudes or a matrix (n, 3) to add points "
            "defined by their longitudes, latitudes and altitudes.");
    }
  }

  /// Insert new data into the search tree
  ///
  /// @param coordinates Coordinates to be copied
  void insert(const pybind11::array_t<Coordinate> &coordinates,
              const pybind11::array_t<Coordinate> &values) {
    detail::check_array_ndim("coordinates", 2, coordinates);
    detail::check_array_ndim("values", 1, values);
    if (coordinates.shape(0) != values.size()) {
      throw std::invalid_argument(
          "coordinates, values could not be broadcast together with shape " +
          detail::ndarray_shape(coordinates) + ", " +
          detail::ndarray_shape(values));
    }
    switch (coordinates.shape(1)) {
      case 2:
        _insert<2>(coordinates, values);
        break;
      case 3:
        _insert<3>(coordinates, values);
        break;
      default:
        throw std::invalid_argument(
            "coordinates must be a matrix (n, 2) to add points defined by "
            "their longitudes and latitudes or a matrix (n, 3) to add points "
            "defined by their longitudes, latitudes and altitudes.");
    }
  }

  /// Search for the nearest K nearest neighbors of a given coordinates.
  auto query(const pybind11::array_t<Type> &coordinates, const uint32_t k,
             const bool within, const size_t num_threads) const
      -> pybind11::tuple {
    detail::check_array_ndim("coordinates", 2, coordinates);
    switch (coordinates.shape(1)) {
      case 2:
        return _query<2>(coordinates, k, within, num_threads);
        break;
      case 3:
        return _query<3>(coordinates, k, within, num_threads);
        break;
      default:
        throw std::invalid_argument(
            "coordinates must be a matrix (n, 2) to search points defined by "
            "their longitudes and latitudes or a matrix(n, 3) to search "
            "points defined by their longitudes, latitudes and altitudes.");
    }
  }

  /// TODO
  auto inverse_distance_weighting(const pybind11::array_t<Type> &coordinates,
                                  distance_t radius, uint32_t k, uint32_t p,
                                  bool within, size_t num_threads) const
      -> pybind11::tuple {
    detail::check_array_ndim("coordinates", 2, coordinates);
    switch (coordinates.shape(1)) {
      case 2:
        return _inverse_distance_weighting<2>(coordinates, radius, k, p, within,
                                              num_threads);
        break;
      case 3:
        return _inverse_distance_weighting<3>(coordinates, radius, k, p, within,
                                              num_threads);
        break;
      default:
        throw std::invalid_argument(
            "coordinates must be a matrix (n, 2) to search points defined by "
            "their longitudes and latitudes or a matrix(n, 3) to search "
            "points defined by their longitudes, latitudes and altitudes.");
    }
  }

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    auto x = pybind11::array_t<Coordinate>(
        pybind11::array::ShapeContainer{{this->size()}});
    auto y = pybind11::array_t<Coordinate>(
        pybind11::array::ShapeContainer{{this->size()}});
    auto z = pybind11::array_t<Coordinate>(
        pybind11::array::ShapeContainer{{this->size()}});
    auto u = pybind11::array_t<Type>(
        pybind11::array::ShapeContainer{{this->size()}});
    auto _x = x.template mutable_unchecked<1>();
    auto _y = y.template mutable_unchecked<1>();
    auto _z = z.template mutable_unchecked<1>();
    auto _u = u.template mutable_unchecked<1>();
    size_t ix = 0;
    std::for_each(this->tree_->begin(), this->tree_->end(),
                  [&](const auto &item) {
                    _x(ix) = boost::geometry::get<0>(item.first);
                    _y(ix) = boost::geometry::get<1>(item.first);
                    _z(ix) = boost::geometry::get<2>(item.first);
                    _u(ix) = item.second;
                    ++ix;
                  });
    auto system = geodetic::System(this->coordinates_.system());
    return pybind11::make_tuple(system.getstate(), x, y, z, u);
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple &state)
      -> RTree<Coordinate, Type> {
    if (state.size() != 5) {
      throw std::runtime_error("invalid state");
    }
    auto system = geodetic::System::setstate(state[0].cast<pybind11::tuple>());
    auto x = state[1].cast<pybind11::array_t<Coordinate>>();
    auto y = state[2].cast<pybind11::array_t<Coordinate>>();
    auto z = state[3].cast<pybind11::array_t<Coordinate>>();
    auto u = state[4].cast<pybind11::array_t<Type>>();

    if (x.size() != y.size() || x.size() != z.size() || x.size() != u.size()) {
      throw std::runtime_error("invalid state");
    }

    auto _x = x.template mutable_unchecked<1>();
    auto _y = y.template mutable_unchecked<1>();
    auto _z = z.template mutable_unchecked<1>();
    auto _u = u.template mutable_unchecked<1>();

    auto vector = std::vector<typename RTree<Coordinate, Type>::value_t>();
    vector.reserve(x.size());

    for (auto ix = 0; ix < x.size(); ++ix) {
      vector.emplace_back(std::make_pair(
          detail::geometry::Point3D<Coordinate>{_x(ix), _y(ix), _z(ix)},
          _u(ix)));
    }
    auto result = RTree<Coordinate, Type>(system);
    static_cast<detail::geometry::RTree<Coordinate, Type, 3>>(result).packing(
        vector);
    return result;
  }

 private:
  /// Packing coordinates
  ///
  /// @param coordinates Coordinates to be copied
  template <size_t Dimensions>
  void _packing(const pybind11::array_t<Coordinate> &coordinates,
                const pybind11::array_t<Type> &values) {
    auto _coordinates = coordinates.template unchecked<2>();
    auto _values = values.template unchecked<1>();
    auto size = coordinates.shape(0);
    auto vector = std::vector<typename RTree<Coordinate, Type>::value_t>();
    auto point = detail::geometry::EquatorialPoint3D<Coordinate>();

    vector.reserve(size);

    for (auto ix = 0; ix < size; ++ix) {
      auto dim = 0ULL;
      for (; dim < Dimensions; ++dim) {
        detail::geometry::point::set(point, _coordinates(ix, dim), dim);
      }
      for (; dim < 3; ++dim) {
        detail::geometry::point::set(point, Coordinate(0), dim);
      }
      vector.emplace_back(
          std::make_pair(this->coordinates_.lla_to_ecef(point), _values(ix)));
    }
    detail::geometry::RTree<Coordinate, Type, 3>::packing(vector);
  }

  /// Insert coordinates
  ///
  /// @param coordinates Coordinates to be copied
  template <size_t Dimensions>
  void _insert(const pybind11::array_t<Coordinate> &coordinates,
               const pybind11::array_t<Type> &values) {
    auto _coordinates = coordinates.template unchecked<2>();
    auto _values = values.template unchecked<1>();
    auto size = coordinates.shape(0);
    auto point = detail::geometry::EquatorialPoint3D<Coordinate>();

    for (auto ix = 0; ix < size; ++ix) {
      auto dim = 0ULL;
      for (; dim < Dimensions; ++dim) {
        detail::geometry::point::set(point, _coordinates(ix, dim), dim);
      }
      for (; dim < 3; ++dim) {
        detail::geometry::point::set(point, Coordinate(0), dim);
      }
      detail::geodetic::RTree<Coordinate, Type>::insert(
          std::make_pair(this->coordinates_.lla_to_ecef(point), _values(ix)));
    }
  }

  /// Search for the nearest K nearest neighbors of a given coordinates.
  template <size_t Dimensions>
  auto _query(const pybind11::array_t<Coordinate> &coordinates,
              const uint32_t k, const bool within,
              const size_t num_threads) const -> pybind11::tuple {
    // Signature of the function of the class to be called.
    using query_t = std::vector<
        typename detail::geodetic::RTree<Coordinate, Type>::result_t> (
        RTree<Coordinate, Type>::*)(
        const detail::geometry::EquatorialPoint3D<Coordinate> &, uint32_t)
        const;

    // Selection of the method performing the calculation.
    const std::function<std::vector<
        typename detail::geodetic::RTree<Coordinate, Type>::result_t>(
        const RTree<Coordinate, Type> &,
        const detail::geometry::EquatorialPoint3D<Coordinate> &, uint32_t)>
        method =
            within
                ? static_cast<query_t>(
                      &detail::geodetic::RTree<Coordinate, Type>::query_within)
                : static_cast<query_t>(
                      &detail::geodetic::RTree<Coordinate, Type>::query);

    auto _coordinates = coordinates.template unchecked<2>();
    auto size = coordinates.shape(0);

    // Allocation of result matrices.
    auto distance = pybind11::array_t<distance_t>(
        pybind11::array::ShapeContainer{size, static_cast<ssize_t>(k)});
    auto value = pybind11::array_t<Type>(
        pybind11::array::ShapeContainer{size, static_cast<ssize_t>(k)});

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
              auto point = detail::geometry::EquatorialPoint3D<Coordinate>();
              for (size_t ix = start; ix < end; ++ix) {
                auto dim = 0ULL;

                for (; dim < Dimensions; ++dim) {
                  detail::geometry::point::set(point, _coordinates(ix, dim),
                                               dim);
                }
                for (; dim < 3; ++dim) {
                  detail::geometry::point::set(point, Coordinate(0), dim);
                }

                auto nearest = method(*this, point, k);
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
  template <size_t Dimensions>
  auto _inverse_distance_weighting(const pybind11::array_t<Type> &coordinates,
                                   distance_t radius, uint32_t k, uint32_t p,
                                   bool within, size_t num_threads) const
      -> pybind11::tuple {
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
              auto point = detail::geometry::EquatorialPoint3D<Coordinate>();
              for (size_t ix = start; ix < end; ++ix) {
                auto dim = 0ULL;

                for (; dim < Dimensions; ++dim) {
                  detail::geometry::point::set(point, _coordinates(ix, dim),
                                               dim);
                }
                for (; dim < 3; ++dim) {
                  detail::geometry::point::set(point, Coordinate(0), dim);
                }

                auto result = detail::geodetic::RTree<
                    Coordinate, Type>::inverse_distance_weighting(point, radius,
                                                                  k, p, within);
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
