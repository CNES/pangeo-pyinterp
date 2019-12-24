// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <memory>
#include <pybind11/numpy.h>
#include "pyinterp/detail/axis.hpp"
#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/math.hpp"

namespace pyinterp {
namespace detail {

/// Builds an Eigen::Vector from a numpy vector
///
/// @param name Variable name
/// @param ndarray Vector to copy
/// @return The numpy vector copied into an Eigen vector.
template <typename T>
inline auto vector_from_numpy(
    const std::string& name,
    pybind11::array_t<T, pybind11::array::c_style>& ndarray)
    -> Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> {
  check_array_ndim(name, 1, ndarray);
  return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(ndarray.mutable_data(),
                                                         ndarray.size());
}

}  // namespace detail

/// Forward declaration
template <typename T>
class Axis;

/// Implementation of the Python wrapper
template <typename T>
class Axis : public detail::Axis<T>,
             public std::enable_shared_from_this<Axis<T>> {
 private:
  /// Opaque marker of a serialized undefined axis.
  static const int64_t UNDEFINED;

  /// Opaque marker of a serialized regular axis.
  static const int64_t REGULAR;

  /// Opaque marker of a serialized irregular axis.
  static const int64_t IRREGULAR;

 public:
  using detail::Axis<T>::Axis;

  /// Create a coordinate axis from values.
  ///
  /// @param points axis values
  /// @param epsilon Maximum allowed difference between two real numbers in
  /// order to consider them equal.
  /// @param is_circle True, if the axis can represent a circle. Be careful,
  /// the angle shown must be expressed in degrees.
  explicit Axis(pybind11::array_t<T, pybind11::array::c_style>& points,
                T epsilon, bool is_circle)
      : Axis<T>(pyinterp::detail::vector_from_numpy("points", points), epsilon,
                is_circle) {}

  /// Get coordinate values.
  ///
  /// @param slice Slice of indexes to read
  /// @return coordinate values
  auto coordinate_values(const pybind11::slice& slice) const
      -> pybind11::array_t<T> {
    size_t start;
    size_t stop;
    size_t step;
    size_t slicelength;

    if (!slice.compute(this->size(), &start, &stop, &step, &slicelength)) {
      throw pybind11::error_already_set();
    }

    auto result = pybind11::array_t<T>(slicelength);
    auto _result = result.template mutable_unchecked<1>();

    {
      pybind11::gil_scoped_release release;
      for (size_t ix = 0; ix < slicelength; ++ix) {
        _result(ix) = (*this)(ix);
      }
    }
    return result;
  }

  /// Given a coordinate position, find what axis element contains it.
  ///
  /// @param coordinate positions in this coordinate system
  /// @param bounded True if you want to obtain the closest value to an index
  ///   outside the axis definition range.
  /// @return A vector containing the indexes corresponding to the nearest
  ///   points on the axis or the value -1 if the *bounded* parameter is set
  ///   to false and the index looked for is located outside the limits of the
  ///   axis.
  auto find_index(const pybind11::array_t<T>& coordinates, bool bounded) const
      -> pybind11::array_t<int64_t> {
    detail::check_array_ndim("coordinates", 1, coordinates);

    auto size = coordinates.size();
    auto result = pybind11::array_t<int64_t>(size);
    auto _result = result.mutable_unchecked<1>();
    auto _coordinates = coordinates.template unchecked<1>();

    {
      pybind11::gil_scoped_release release;
      for (pybind11::ssize_t ix = 0; ix < size; ++ix) {
        _result(ix) = detail::Axis<T>::find_index(_coordinates(ix), bounded);
      }
    }
    return result;
  }

  /// Get a tuple that fully encodes the state of this instance
  auto getstate() const -> pybind11::tuple {
    // Regular
    {
      auto ptr = dynamic_cast<detail::axis::container::Regular<T>*>(
          this->handler().get());
      if (ptr != nullptr) {
        return pybind11::make_tuple(REGULAR, ptr->front(), ptr->back(),
                                    ptr->size(), this->is_circle());
      }
    }
    // Irregular
    {
      auto ptr = dynamic_cast<detail::axis::container::Irregular<T>*>(
          this->handler().get());
      if (ptr != nullptr) {
        auto values = pybind11::array_t<T>(ptr->size());
        auto _values = values.template mutable_unchecked<1>();
        for (auto ix = 0LL; ix < ptr->size(); ++ix) {
          _values[ix] = ptr->coordinate_value(ix);
        }
        return pybind11::make_tuple(IRREGULAR, values, this->is_circle());
      }
    }
    // Undefined
    auto ptr = dynamic_cast<detail::axis::container::Undefined<T>*>(
        this->handler().get());
    if (ptr != nullptr) {
      return pybind11::make_tuple(UNDEFINED);
    }
    throw std::runtime_error("unknown axis handler");
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple& state) -> Axis<T> {
    if (state.empty()) {
      throw std::invalid_argument("invalid state");
    }
    auto identification = state[0].cast<int64_t>();
    switch (identification) {
      case UNDEFINED:
        return Axis();
        break;
      case IRREGULAR: {
        auto ndarray = state[1].cast<pybind11::array_t<T>>();
        return Axis(std::shared_ptr<detail::axis::container::Abstract<T>>(
                        new detail::axis::container::Irregular<T>(
                            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(
                                ndarray.mutable_data(), ndarray.size()))),
                    state[2].cast<bool>());
      }
      case REGULAR:
        return Axis(std::shared_ptr<detail::axis::container::Abstract<T>>(
                        new detail::axis::container::Regular<T>(
                            state[1].cast<T>(), state[2].cast<T>(),
                            state[3].cast<T>())),
                    state[4].cast<bool>());
      default:
        throw std::invalid_argument("invalid state");
    }
  }
};

template <typename T>
const int64_t Axis<T>::UNDEFINED = 0x618d86f8334b6c93;

template <typename T>
const int64_t Axis<T>::REGULAR = 0x22d06666a82610a3;

template <typename T>
const int64_t Axis<T>::IRREGULAR = 0x3ab687f709def680;

}  // namespace pyinterp
