// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "pyinterp/axis.hpp"
#include "pyinterp/dateutils.hpp"
#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp {
namespace detail {

/// Check the properties of the numpy array and returns the type of dates
/// encoded in this parameter.
inline auto dtype(const std::string &name, const pybind11::array &array)
    -> dateutils::DType {
  detail::check_array_ndim(name, 1, array);
  if ((array.flags() & pybind11::array::c_style) == 0) {
    throw pybind11::type_error(name + " must be a C-style contiguous array");
  }
  return dateutils::DType(static_cast<std::string>(
      pybind11::str(static_cast<pybind11::handle>(array.dtype()))));
}

/// Get the numpy data type and the raw values stored in a numpy
/// datetime64/timedelta64 array
inline auto vector_from_numpy(const pybind11::array &array)
    -> Eigen::Map<const Vector<int64_t>> {
  auto ptr = array.unchecked<int64_t, 1>();
  return {&ptr[0], array.size()};
}

}  // namespace detail

/// Forward declaration
class TemporalAxis;

/// Implementation of the Python wrapper
///
/// @tparam T Type of data handled by the axis.
class TemporalAxis : public Axis<int64_t> {
 public:
  using Axis<int64_t>::Axis;

  /// Create a coordinate axis from values.
  ///
  /// @param points axis values
  explicit TemporalAxis(const pybind11::array &points)
      : TemporalAxis(Axis<int64_t>(detail::vector_from_numpy(points), 0, false),
                     detail::dtype("points", points)) {}

  /// Returns the numpy dtype handled by the axis.
  [[nodiscard]] inline auto dtype() const -> pybind11::dtype {
    return pybind11::dtype(static_cast<std::string>(dtype_));
  }

  /// @copydoc detail::Axis::coordinate_value(const int64_t) const
  [[nodiscard]] inline auto coordinate_value(const int64_t index) const
      -> pybind11::array {
    return as_datetype(
        [&]() -> int64_t { return Axis<int64_t>::coordinate_value(index); },
        dtype_);
  }

  /// @copydoc detail::Axis::font() const
  [[nodiscard]] inline auto front() const -> pybind11::array {
    return as_datetype([&]() -> int64_t { return Axis<int64_t>::front(); },
                       dtype_);
  }

  /// @copydoc detail::Axis::back() const
  [[nodiscard]] inline auto back() const -> pybind11::array {
    return as_datetype([&]() -> int64_t { return Axis<int64_t>::back(); },
                       dtype_);
  }

  /// @copydoc detail::Axis::increment() const
  [[nodiscard]] inline auto increment() const -> pybind11::array {
    return as_datetype([&]() -> int64_t { return Axis<int64_t>::increment(); },
                       dtype_.as_timedelta64());
  }

  /// @copydoc detail::Axis::max_value() const
  [[nodiscard]] inline auto max_value() const -> pybind11::array {
    return as_datetype([&]() -> int64_t { return Axis<int64_t>::max_value(); },
                       dtype_);
  }

  /// @copydoc detail::Axis::min_value() const
  [[nodiscard]] inline auto min_value() const -> pybind11::array {
    return as_datetype([&]() -> int64_t { return Axis<int64_t>::min_value(); },
                       dtype_);
  }

  /// @copydoc Axis::coordinates_values(const pybind11::slice&) const
  auto coordinate_values(const pybind11::slice &slice) const
      -> pybind11::array {
    size_t start;
    size_t stop;
    size_t step;
    size_t slicelength;

    if (!slice.compute(this->size(), &start, &stop, &step, &slicelength)) {
      throw pybind11::error_already_set();
    }

    auto result =
        pybind11::array(dtype(), pybind11::array::ShapeContainer{slicelength},
                        pybind11::array::ShapeContainer{}, nullptr);
    auto _result = result.template mutable_unchecked<int64_t, 1>();

    {
      pybind11::gil_scoped_release release;
      for (size_t ix = 0; ix < slicelength; ++ix) {
        _result(ix) = (*this)(static_cast<int64_t>(start));
        start += step;
      }
    }
    return result;
  }

  /// @copydoc Axis::find_index(const pybind11::array_t<int64_t>, bool) const
  auto find_index(const pybind11::array &coordinates, bool bounded) const
      -> pybind11::array_t<int64_t> {
    const auto values = safe_cast("coordinates", coordinates);
    const auto size = values.size();
    auto result = pybind11::array_t<int64_t>(size);
    auto _result = result.mutable_unchecked<1>();
    auto _values = values.template unchecked<int64_t, 1>();

    {
      pybind11::gil_scoped_release release;
      for (pybind11::ssize_t ix = 0; ix < size; ++ix) {
        _result(ix) =
            detail::Axis<int64_t>::find_nearest_index(_values(ix), bounded);
      }
    }
    return result;
  }

  /// @copydoc Axis::find_indexes(const pybind11::array_t<int64_t>) const
  auto find_indexes(const pybind11::array &coordinates) const
      -> pybind11::array_t<int64_t> {
    const auto values = safe_cast("coordinates", coordinates);
    const auto size = values.size();
    auto result =
        pybind11::array_t<int64_t>(pybind11::array::ShapeContainer({size, 2}));
    auto _result = result.mutable_unchecked<2>();
    auto _values = values.template unchecked<int64_t, 1>();

    {
      pybind11::gil_scoped_release release;
      for (pybind11::ssize_t ix = 0; ix < size; ++ix) {
        auto indexes = detail::Axis<int64_t>::find_indexes(_values(ix));
        if (indexes) {
          std::tie(_result(ix, 0), _result(ix, 1)) = *indexes;
        } else {
          _result(ix, 0) = _result(ix, 1) = -1;
        }
      }
    }
    return result;
  }

  /// Convert the dates of the vector in the same unit as the time axis
  /// defined in this instance.
  [[nodiscard]] auto safe_cast(const std::string &name,
                               const pybind11::array &coordinates) const
      -> pybind11::array {
    auto dtype = detail::dtype(name, coordinates);

    if (dtype.datetype() != dtype_.datetype()) {
      throw std::runtime_error("Cannot cast " + name + " to " +
                               static_cast<std::string>(dtype) +
                               " because the time axis is defined in " +
                               static_cast<std::string>(dtype_));
    }

    if (dtype_ < dtype) {
      auto message = "implicit conversion turns " +
                     static_cast<std::string>(dtype) + " into " +
                     static_cast<std::string>(dtype_);
      if (PyErr_WarnEx(PyExc_UserWarning, message.c_str(), 1) == -1) {
        throw pybind11::error_already_set();
      }
    }
    if (dtype_ == dtype) {
      return coordinates;
    }
    return coordinates.attr("astype")(static_cast<std::string>(dtype_));
  }

  /// @copydoc detail::Axis::getstate() const
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(Axis<int64_t>::getstate(),
                                static_cast<std::string>(dtype_));
  }

  /// @copydoc detail::Axis::setstate(const pybind11::tuple&) const
  static auto setstate(const pybind11::tuple &state) -> TemporalAxis {
    if (state.size() != 2) {
      throw std::invalid_argument("invalid state");
    }
    return {Axis<int64_t>::setstate(state[0].cast<pybind11::tuple>()),
            dateutils::DType(state[1].cast<std::string>())};
  }

  /// @copydoc detail::Axis::coordinate_repr(const int64_t) const
  inline auto coordinate_repr(const int64_t value) const
      -> std::string override {
    return dateutils::datetime64_to_string(value, dtype_);
  }

  /// @copydoc detail::Axis::operator std::string() const
  explicit operator std::string() const override {
    auto ss = std::stringstream();
    ss << "<pyinterp.core.TemporalAxis>" << std::endl;
    if (is_regular()) {
      ss << "  min_value: "
         << static_cast<std::string>(pybind11::str(min_value())) << std::endl;
      ss << "  max_value: "
         << static_cast<std::string>(pybind11::str(max_value())) << std::endl;
      ss << "  step     : "
         << static_cast<std::string>(pybind11::str(increment()));
    } else {
      auto values = coordinate_values(pybind11::slice(0, size(), 1));
      ss << "  values   : "
         << detail::pad(static_cast<std::string>(pybind11::str(values)));
    }
    return ss.str();
  }

 private:
  dateutils::DType dtype_;

  /// Construct a new instance from the base class.
  TemporalAxis(Axis<int64_t> base, const dateutils::DType &dtype)
      : Axis<int64_t>(std::move(base)), dtype_(dtype) {}

  /// Returns the result of a method wrapped in a pybind11::array.
  template <typename Getter>
  inline auto as_datetype(const Getter &getter,
                          const dateutils::DType &dtype) const
      -> pybind11::array {
    auto value = getter();
    return {pybind11::dtype(static_cast<std::string>(dtype)), {}, {}, &value};
  }
};

}  // namespace pyinterp
