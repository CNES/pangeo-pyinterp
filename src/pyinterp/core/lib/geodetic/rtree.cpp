// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geodetic/rtree.hpp"

namespace pyinterp::geodetic {

template <typename Point, typename Strategy, typename T>
class RBF : public detail::math::RBF<T> {
 public:
  RBF(const Strategy &strategy, const T &epsilon, const T &smooth,
      const detail::math::RadialBasisFunction rbf)
      : detail::math::RBF<T>(epsilon, smooth, rbf), strategy_(strategy) {}

  [[nodiscard]] inline auto calculate_distance(
      const Eigen::Ref<const Vector<T>> &x,
      const Eigen::Ref<const Vector<T>> &y) const -> T override {
    return boost::geometry::distance(Point(x(0), x(1)), Point(y(0), y(1)),
                                     strategy_);
  }

 private:
  const Strategy &strategy_;
};

RTree::RTree(const std::optional<detail::geodetic::Spheroid> &wgs) : base_t() {
  auto spheroid = wgs.value_or(detail::geodetic::Spheroid());
  strategy_ = strategy_t(boost::geometry::srs::spheroid<double>(
      spheroid.semi_major_axis(), spheroid.semi_minor_axis()));
}

auto RTree::equatorial_bounds() const -> std::optional<Box> {
  auto box = this->bounds();
  if (box.has_value()) {
    return Box(Point(boost::geometry::get<0>(box.value().min_corner()),
                     boost::geometry::get<0>(box.value().min_corner())),
               Point(boost::geometry::get<0>(box.value().max_corner()),
                     boost::geometry::get<0>(box.value().max_corner())));
  }
  return {};
}

auto RTree::packing(const Eigen::Ref<const Vector<double>> &lon,
                    const Eigen::Ref<const Vector<double>> &lat,
                    const Eigen::Ref<const Vector<double>> &values) -> void {
  detail::check_container_size("lon", lon, "lat", lat, "values", values);

  auto vector = std::vector<RTree::value_t>();
  vector.reserve(lon.size());

  auto _x = lon.data();
  auto _y = lat.data();
  auto _z = values.data();

  for (auto ix = 0; ix < lon.size(); ++ix) {
    vector.emplace_back(std::make_pair(
        point_t(detail::math::normalize_angle(*_x, -180.0, 360.0), *_y), *_z));
    ++_x;
    ++_y;
    ++_z;
  }
  base_t::packing(vector);
}

/// Insert new data into the search tree
///
auto RTree::insert(const Eigen::Ref<const Vector<double>> &lon,
                   const Eigen::Ref<const Vector<double>> &lat,
                   const Eigen::Ref<const Vector<double>> &values) -> void {
  detail::check_container_size("lon", lon, "lat", lat, "values", values);

  auto _x = lon.data();
  auto _y = lat.data();
  auto _z = values.data();

  for (auto ix = 0; ix < lon.size(); ++ix) {
    base_t::insert(std::make_pair(
        point_t(detail::math::normalize_angle(*_x, -180.0, 360.0), *_y), *_z));
    ++_x;
    ++_y;
    ++_z;
  }
}

auto RTree::query(const Eigen::Ref<const Vector<double>> &lon,
                  const Eigen::Ref<const Vector<double>> &lat, const uint32_t k,
                  const bool within, const size_t num_threads) const
    -> pybind11::tuple {
  detail::check_container_size("lon", lon, "lat", lat);
  auto size = lon.size();

  auto requester = within ? static_cast<Requester>(&base_t::query_within)
                          : static_cast<Requester>(&base_t::query);

  // Allocation of result matrices.
  auto distance = pybind11::array_t<distance_t>(
      pybind11::array::ShapeContainer{size, static_cast<pybind11::ssize_t>(k)});
  auto value = pybind11::array_t<double>(
      pybind11::array::ShapeContainer{size, static_cast<pybind11::ssize_t>(k)});

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
            for (size_t ix = start; ix < end; ++ix) {
              auto nearest = std::invoke(
                  requester, static_cast<base_t>(*this),
                  point_t(detail::math::normalize_angle(lon(ix), -180.0, 360.0),
                          lat(ix)),
                  strategy_, k);
              auto jx = 0ULL;

              // Fill in the calculation result for all neighbors found
              for (; jx < nearest.size(); ++jx) {
                _distance(ix, jx) = nearest[jx].first;
                _value(ix, jx) = nearest[jx].second;
              }

              // The rest of the result is filled with invalid values.
              for (; jx < k; ++jx) {
                _distance(ix, jx) = -1;
                _value(ix, jx) = -1;
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

auto RTree::inverse_distance_weighting(
    const Eigen::Ref<const Vector<double>> &lon,
    const Eigen::Ref<const Vector<double>> &lat,
    const std::optional<double> &radius, const uint32_t k, const uint32_t p,
    const bool within, const size_t num_threads) const -> pybind11::tuple {
  detail::check_container_size("lon", lon, "lat", lat);
  auto _radius = radius.value_or(std::numeric_limits<double>::max());
  auto size = lon.size();

  // Allocation of result vectors.
  auto data = pybind11::array_t<double>(pybind11::array::ShapeContainer{size});
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
            for (size_t ix = start; ix < end; ++ix) {
              auto result = base_t::inverse_distance_weighting(
                  point_t(detail::math::normalize_angle(lon(ix), -180.0, 360.0),
                          lat(ix)),
                  strategy_, _radius, k, p, within);
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

auto RTree::radial_basis_function(const Eigen::Ref<const Vector<double>> &lon,
                                  const Eigen::Ref<const Vector<double>> &lat,
                                  const std::optional<double> &radius,
                                  const uint32_t k,
                                  const detail::math::RadialBasisFunction rbf,
                                  const std::optional<double> &epsilon,
                                  const double smooth, const bool within,
                                  const size_t num_threads) const
    -> pybind11::tuple {
  detail::check_container_size("lon", lon, "lat", lat);
  auto _radius = radius.value_or(std::numeric_limits<double>::max());
  auto size = lon.size();

  // Construction of the interpolator.
  auto rbf_handler = RBF<point_t, strategy_t, double>(
      strategy_,
      epsilon.value_or(std::numeric_limits<promotion_t>::quiet_NaN()), smooth,
      rbf);

  // Allocation of result vectors.
  auto data = pybind11::array_t<double>(pybind11::array::ShapeContainer{size});
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
            for (size_t ix = start; ix < end; ++ix) {
              auto result = base_t::radial_basis_function(
                  point_t(detail::math::normalize_angle(lon(ix), -180.0, 360.0),
                          lat(ix)),
                  strategy_, rbf_handler, _radius, k, within);
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

auto RTree::window_function(const Eigen::Ref<const Vector<double>> &lon,
                            const Eigen::Ref<const Vector<double>> &lat,
                            const double radius, const uint32_t k,
                            const detail::math::window::Function wf,
                            const std::optional<double> &arg, bool within,
                            const size_t num_threads) const -> pybind11::tuple {
  detail::check_container_size("lon", lon, "lat", lat);
  auto _arg = arg.value_or(0);
  auto size = lon.size();

  // Allocation of result vectors.
  auto data = pybind11::array_t<double>(pybind11::array::ShapeContainer{size});
  auto neighbors =
      pybind11::array_t<uint32_t>(pybind11::array::ShapeContainer{size});

  auto wf_handler = detail::math::WindowFunction<double>(wf);

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
            for (size_t ix = start; ix < end; ++ix) {
              auto result = base_t::window_function(
                  point_t(detail::math::normalize_angle(lon(ix), -180.0, 360.0),
                          lat(ix)),
                  strategy_, wf_handler, _arg, radius, k, within);
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

[[nodiscard]] auto RTree::getstate() const -> pybind11::tuple {
  auto coordinates = pybind11::array_t<double>(pybind11::array::ShapeContainer{
      {static_cast<pybind11::ssize_t>(this->size()), 2}});
  auto data = pybind11::array_t<double>(pybind11::array::ShapeContainer{
      {static_cast<pybind11::ssize_t>(this->size())}});
  auto _coordinates = coordinates.template mutable_unchecked<2>();
  auto _data = data.template mutable_unchecked<1>();
  size_t ix = 0;
  std::for_each(this->tree_->begin(), this->tree_->end(),
                [&](const auto &item) {
                  _coordinates(ix, 0) = boost::geometry::get<0>(item.first);
                  _coordinates(ix, 1) = boost::geometry::get<1>(item.first);
                  _data(ix) = item.second;
                  ++ix;
                });
  const auto &model = strategy_.model();
  auto a = model.template get_radius<0>();
  auto b = model.template get_radius<1>();
  return pybind11::make_tuple(a, b, coordinates, data);
}

/// Create a new instance from a registered state of an instance of this
/// object.
auto RTree::setstate(const pybind11::tuple &state) -> RTree {
  if (state.size() != 4) {
    throw std::runtime_error("invalid state");
  }
  auto a = state[0].cast<double>();
  auto b = state[1].cast<double>();
  auto coordinates = state[2].cast<pybind11::array_t<double>>();
  auto data = state[3].cast<pybind11::array_t<double>>();

  if (coordinates.shape(1) != 2 || coordinates.shape(0) != data.size()) {
    throw std::runtime_error("invalid state");
  }

  auto _coordinates = coordinates.template mutable_unchecked<2>();
  auto _data = data.template mutable_unchecked<1>();

  auto vector = std::vector<typename RTree::value_t>();
  vector.reserve(data.size());

  for (auto ix = 0; ix < data.size(); ++ix) {
    vector.emplace_back(std::make_pair(
        point_t(_coordinates(ix, 0), _coordinates(ix, 1)), _data(ix)));
  }
  auto result = RTree();
  result.strategy_ = strategy_t({a, b});
  static_cast<detail::geometry::RTree<RTree::point_t, double>>(result).packing(
      vector);
  return result;
}

}  // namespace pyinterp::geodetic
