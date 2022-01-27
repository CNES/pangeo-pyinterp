// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <regex>
#include <sstream>
#include <string>

#include "pyinterp/detail/axis.hpp"
#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/math.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp {
namespace detail {
namespace axis {
/// Opaque marker of a serialized undefined axis.
constexpr int64_t UNDEFINED = 0x618d86f8334b6c93;
/// Opaque marker of a serialized regular axis.
constexpr int64_t REGULAR = 0x22d06666a82610a3;
/// Opaque marker of a serialized irregular axis.
constexpr int64_t IRREGULAR = 0x3ab687f709def680;
}  // namespace axis

/// Builds an Eigen::Vector from a numpy vector
///
/// @param name Variable name
/// @param ndarray Vector to copy
/// @return The numpy vector copied into an Eigen vector.
template <typename T>
inline auto vector_from_numpy(
    const std::string &name,
    const pybind11::array_t<T, pybind11::array::c_style> &ndarray)
    -> Eigen::Map<const Vector<T>> {
  check_array_ndim(name, 1, ndarray);
  return Eigen::Map<const Vector<T>>(ndarray.data(), ndarray.size());
}

/// Pad a string with spaces
inline auto pad(const std::string &str) -> std::string {
  auto re = std::regex("(?:\r\n|\n)");
  auto first = std::sregex_token_iterator(str.begin(), str.end(), re, -1);
  auto last = std::sregex_token_iterator();
  auto ss = std::stringstream{};
  for (auto it = first; it != last; ++it) {
    if (it != first) {
      ss << std::endl << "             ";
    }
    ss << *it;
  }
  return ss.str();
}

}  // namespace detail

/// Forward declaration
///
/// @tparam T Type of data handled by the axis.
template <typename T>
class Axis;

/// Implementation of the Python wrapper
///
/// @tparam T Type of data handled by the axis.
template <typename T>
class Axis : public detail::Axis<T>,
             public std::enable_shared_from_this<Axis<T>> {
 public:
  using detail::Axis<T>::Axis;
  using detail::Axis<T>::find_indexes;

  /// Create a coordinate axis from values.
  ///
  /// @param points axis values
  /// @param epsilon Maximum allowed difference between two real numbers in
  /// order to consider them equal.
  /// @param is_circle True, if the axis can represent a circle. Be careful,
  /// the angle shown must be expressed in degrees.
  explicit Axis(const pybind11::array_t<T, pybind11::array::c_style> &points,
                T epsilon, bool is_circle)
      : Axis<T>(pyinterp::detail::vector_from_numpy("points", points), epsilon,
                is_circle) {}

  /// Get coordinate values.
  ///
  /// @param slice Slice of indexes to read
  /// @return coordinate values
  auto coordinate_values(const pybind11::slice &slice) const
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
        _result(ix) = (*this)(start);
        start += step;
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
  auto find_index(const pybind11::array_t<T> &coordinates, bool bounded) const
      -> pybind11::array_t<int64_t> {
    detail::check_array_ndim("coordinates", 1, coordinates);

    auto size = coordinates.size();
    auto result = pybind11::array_t<int64_t>(size);
    auto _result = result.mutable_unchecked<1>();
    auto _coordinates = coordinates.template unchecked<1>();

    {
      pybind11::gil_scoped_release release;
      for (pybind11::ssize_t ix = 0; ix < size; ++ix) {
        _result(ix) =
            detail::Axis<T>::find_nearest_index(_coordinates(ix), bounded);
      }
    }
    return result;
  }

  /// Given a coordinate position, find grids elements around it.
  /// This mean that
  /// @code
  /// (*this)(i0) <= coordinate < (*this)(i1)
  /// @endcode
  ///
  /// @param coordinate position in this coordinate system
  /// @return A matrix of shape (n, 2). The first column of the matrix
  /// contains the indexes i0 and the second column the indexes i1
  /// found.
  auto find_indexes(const pybind11::array_t<T> &coordinates) const
      -> pybind11::array_t<int64_t> {
    detail::check_array_ndim("coordinates", 1, coordinates);

    auto size = coordinates.size();
    auto result =
        pybind11::array_t<int64_t>(pybind11::array::ShapeContainer({size, 2}));
    auto _result = result.mutable_unchecked<2>();
    auto _coordinates = coordinates.template unchecked<1>();

    {
      pybind11::gil_scoped_release release;
      for (pybind11::ssize_t ix = 0; ix < size; ++ix) {
        auto indexes = detail::Axis<T>::find_indexes(_coordinates(ix));
        if (indexes) {
          std::tie(_result(ix, 0), _result(ix, 1)) = *indexes;
        } else {
          _result(ix, 0) = _result(ix, 1) = -1;
        }
      }
    }
    return result;
  }

  /// @copydoc detail::Axis::operator std::string() const
  virtual explicit operator std::string() const {
    auto ss = std::stringstream();
    ss << "<pyinterp.core.Axis>" << std::endl;
    if (this->is_regular()) {
      ss << "  min_value: " << this->min_value() << std::endl;
      ss << "  max_value: " << this->max_value() << std::endl;
      ss << "  step     : " << this->increment() << std::endl;
    } else {
      auto values = coordinate_values(pybind11::slice(0, this->size(), 1));
      ss << "  values   : "
         << detail::pad(static_cast<std::string>(pybind11::str(values)))
         << std::endl;
    }
    ss << "  is_circle: " << std::boolalpha << this->is_circle();
    return ss.str();
  }

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    // Regular
    {
      auto ptr = dynamic_cast<detail::axis::container::Regular<T> *>(
          this->handler().get());
      if (ptr != nullptr) {
        return pybind11::make_tuple(detail::axis::REGULAR, ptr->front(),
                                    ptr->back(), ptr->size(),
                                    this->is_circle());
      }
    }
    // Irregular
    {
      auto ptr = dynamic_cast<detail::axis::container::Irregular<T> *>(
          this->handler().get());
      if (ptr != nullptr) {
        auto values = pybind11::array_t<T>(ptr->size());
        auto _values = values.template mutable_unchecked<1>();
        for (auto ix = 0LL; ix < ptr->size(); ++ix) {
          _values[ix] = ptr->coordinate_value(ix);
        }
        return pybind11::make_tuple(detail::axis::IRREGULAR, values,
                                    this->is_circle());
      }
    }
    // Undefined
    auto ptr = dynamic_cast<detail::axis::container::Undefined<T> *>(
        this->handler().get());
    if (ptr != nullptr) {
      return pybind11::make_tuple(detail::axis::UNDEFINED);
    }
    throw std::runtime_error("unknown axis handler");
  }

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple &state) -> Axis<T> {
    if (state.empty()) {
      throw std::invalid_argument("invalid state");
    }
    auto identification = state[0].cast<int64_t>();
    switch (identification) {
      case detail::axis::UNDEFINED:
        return Axis();
        break;
      case detail::axis::IRREGULAR: {
        auto ndarray = state[1].cast<pybind11::array_t<T>>();
        return Axis(
            std::shared_ptr<detail::axis::container::Abstract<T>>(
                new detail::axis::container::Irregular<T>(Eigen::Map<Vector<T>>(
                    ndarray.mutable_data(), ndarray.size()))),
            state[2].cast<bool>());
      }
      case detail::axis::REGULAR:
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

}  // namespace pyinterp
