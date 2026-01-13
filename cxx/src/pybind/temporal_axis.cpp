// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/temporal_axis.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <ranges>
#include <utility>

#include "pyinterp/math/temporal_axis.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"
#include "pyinterp/pybind/numpy.hpp"

namespace nb = nanobind;

namespace pyinterp::pybind {

// Convert timedelta64 scalar to int64_t with resolution validation
// @param[in] param_name Name of the parameter (for error messages)
// @param[in] timedelta Numpy timedelta64 scalar
// @param[in] target_dtype Target dtype for conversion
// @return Converted int64_t value or std::nullopt if input is None
inline auto convert_timedelta64(const std::string &param_name,
                                const nb::object &timedelta,
                                const dateutils::DType &target_dtype)
    -> std::optional<int64_t> {
  if (timedelta.is_none()) {
    return std::nullopt;
  }

  // If the object has no dtype attribute, it's not a numpy object
  if (!nb::hasattr(timedelta, "dtype")) {
    throw std::invalid_argument(param_name +
                                " must be a numpy.timedelta64 scalar or None");
  }

  // Retrieve and validate the dtype of the input timedelta64
  auto cxx_dtype = retrieve_dtype(param_name, timedelta);
  if (cxx_dtype.datetype() != dateutils::DType::DateType::kTimedelta64) {
    throw std::invalid_argument(param_name +
                                " must be a numpy.timedelta64 scalar");
  }

  // The input resolution must not be finer than the target resolution
  if (cxx_dtype > target_dtype.as_timedelta64()) {
    throw std::invalid_argument(
        std::format("{} resolution {} is finer than points resolution {}",
                    param_name, static_cast<std::string>(cxx_dtype),
                    static_cast<std::string>(target_dtype.as_timedelta64())));
  }

  // Input resolution is coarser than target resolution: issue a warning
  if (cxx_dtype < target_dtype.as_timedelta64()) {
    auto msg =
        std::format("converting {} from {} to {}", param_name,
                    static_cast<std::string>(cxx_dtype),
                    static_cast<std::string>(target_dtype.as_timedelta64()));
    if (PyErr_WarnEx(PyExc_RuntimeWarning, msg.c_str(), 2) == -1) {
      nb::raise_python_error();
    }
  }

  // Convert scalar to int64 via .view() then extract the value with item()
  auto data = nb::cast<int64_t>(timedelta.attr("view")("int64").attr("item")());
  // Convert to target resolution if needed
  return dateutils::convert(data, cxx_dtype, target_dtype.as_timedelta64());
}

// Convert input temporal array to internal resolution (int64_t)
// @param[in] name Name of the parameter (for error messages)
// @param[in] coordinates Numpy array of datetime64/timedelta64 values
// @param[in] target The target axis dtype for conversion
// @return Eigen vector of int64_t values converted to axis resolution
inline auto cast_to_axis_resolution(const std::string &name,
                                    const nb::object &coordinates,
                                    const dateutils::DType &target)
    -> Vector<int64_t> {
  auto source = retrieve_dtype(name, coordinates);

  if (source.datetype() != target.datetype()) {
    throw std::invalid_argument(
        std::format("{} must be of type {}, got {}", name,
                    static_cast<std::string>(target.datetype_name()),
                    static_cast<std::string>(source.datetype_name())));
  }

  auto result = numpy_to_vector(coordinates);
  // If the source resolution is different from the target resolution, perform
  // conversion
  if (source != target) {
    dateutils::convert(result, source, target);
  }
  return result;
}

TemporalAxis::TemporalAxis(const nb::object &points, const nb::object &epsilon,
                           const nb::object &period)
    : math::TemporalAxis() {
  auto dtype = retrieve_dtype("points", points);
  auto mapped_integer_values = numpy_to_vector(points);
  auto integer_epsilon =
      convert_timedelta64("epsilon", epsilon, dtype).value_or(0);
  auto integer_period = convert_timedelta64("period", period, dtype);
  {
    nb::gil_scoped_release release;
    auto self = math::TemporalAxis(dtype, mapped_integer_values,
                                   integer_epsilon, integer_period);
    new (this) TemporalAxis(std::move(self));
  }
}

auto TemporalAxis::dtype() const -> nb::object {
  return to_dtype(math::TemporalAxis::dtype());
}

auto TemporalAxis::front() const -> nb::object {
  return make_scalar(math::TemporalAxis::front(), math::TemporalAxis::dtype());
}

auto TemporalAxis::back() const -> nb::object {
  return make_scalar(math::TemporalAxis::back(), math::TemporalAxis::dtype());
}

auto TemporalAxis::min_value() const -> nb::object {
  return make_scalar(math::TemporalAxis::min_value(),
                     math::TemporalAxis::dtype());
}

auto TemporalAxis::max_value() const -> nb::object {
  return make_scalar(math::TemporalAxis::max_value(),
                     math::TemporalAxis::dtype());
}

auto TemporalAxis::increment() const -> nb::object {
  return make_timedelta64_scalar(math::TemporalAxis::increment(),
                                 math::TemporalAxis::dtype().as_timedelta64());
}

auto TemporalAxis::period() const -> nanobind::object {
  auto period = math::TemporalAxis::period();
  if (period.has_value()) {
    return make_timedelta64_scalar(
        period.value(), math::TemporalAxis::dtype().as_timedelta64());
  }
  return nanobind::none();
}

auto TemporalAxis::coordinate_value(int64_t index) const -> nanobind::object {
  return make_scalar(math::TemporalAxis::coordinate_value(index),
                     math::TemporalAxis::dtype());
}

auto TemporalAxis::coordinate_values(const nanobind::slice &slice) const
    -> nanobind::object {
  auto [start, stop, step, slicelength] = slice.compute(size());

  Vector<int64_t> values(slicelength);
  {
    nb::gil_scoped_release release;

    for (int64_t ix = 0; std::cmp_less(ix, slicelength); ++ix) {
      values[ix] = math::TemporalAxis::coordinate_value(
          static_cast<int64_t>(start + ix * step));
    }
  }
  return vector_to_numpy(std::move(values), math::TemporalAxis::dtype());
}

auto TemporalAxis::find_index(const nanobind::object &coordinates,
                              bool bounded) const -> Vector<int64_t> {
  auto integer_coordinates = cast_to_axis_resolution(
      "coordinates", coordinates, math::TemporalAxis::dtype());
  auto size = integer_coordinates.size();
  auto result = Vector<int64_t>(size);

  {
    nb::gil_scoped_release release;

    for (auto ix : std::views::iota(int64_t{0}, static_cast<int64_t>(size))) {
      result(ix) =
          math::TemporalAxis::find_index(integer_coordinates(ix), bounded);
    }
  }
  return result;
}

auto TemporalAxis::find_indexes(const nanobind::object &coordinates) const
    -> Eigen::Matrix<int64_t, Eigen::Dynamic, 2, Eigen::RowMajor> {
  auto integer_coordinates = cast_to_axis_resolution(
      "coordinates", coordinates, math::TemporalAxis::dtype());
  auto size = integer_coordinates.size();
  auto result =
      Eigen::Matrix<int64_t, Eigen::Dynamic, 2, Eigen::RowMajor>(size, 2);

  {
    nb::gil_scoped_release release;

    for (auto ix : std::views::iota(int64_t{0}, size)) {
      if (auto indexes =
              math::TemporalAxis::find_indexes(integer_coordinates(ix))) {
        const auto [i0, i1] = *indexes;
        result(ix, 0) = i0;
        result(ix, 1) = i1;
      } else {
        result.row(ix).setConstant(-1);
      }
    }
  }
  return result;
}

auto TemporalAxis::cast_to_int64(const nanobind::object &array) const
    -> Vector<int64_t> {
  const auto &dtype = math::TemporalAxis::dtype();
  return cast_to_axis_resolution("array", array, dtype);
}

auto TemporalAxis::cast_to_temporal_axis(const nanobind::object &array) const
    -> nanobind::object {
  return vector_to_numpy(cast_to_int64(array), math::TemporalAxis::dtype());
}

auto TemporalAxis::getstate() const -> nb::tuple {
  serialization::Writer state;
  {
    nanobind::gil_scoped_release release;
    state = math::TemporalAxis::pack();
  }
  return nanobind::make_tuple(writer_to_ndarray(std::move(state)));
}

auto TemporalAxis::setstate(const nb::tuple &state) -> TemporalAxis {
  if (state.size() != 1) {
    throw std::invalid_argument("Invalid state");
  }
  auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
  auto reader = reader_from_ndarray(array);
  {
    nanobind::gil_scoped_release release;
    return TemporalAxis(math::TemporalAxis::unpack(reader));
  }
}

}  // namespace pyinterp::pybind
