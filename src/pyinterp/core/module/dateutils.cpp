// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/dateutils.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
// clang-format off
#include <datetime.h>
// clang-format on
#include <iomanip>
#include <sstream>

namespace py = pybind11;
namespace dateutils = pyinterp::dateutils;

namespace pyinterp::dateutils {

const std::regex DType::pattern_(
    R"((datetime64|timedelta64)\[(Y|M|W|D|h|m|s|(?:[munpfa]s))\])",
    std::regex::optimize);

}

namespace detail {

static auto fractional_seconds_from_dtype(const pybind11::dtype &dtype)
    -> dateutils::FractionalSeconds {
  auto type_num =
      pybind11::detail::array_descriptor_proxy(dtype.ptr())->type_num;
  if (type_num != 21 /* NPY_DATETIME */) {
    throw std::invalid_argument(
        "array must be a numpy array of datetime64 items");
  }
  return dateutils::FractionalSeconds(static_cast<std::string>(
      pybind11::str(static_cast<pybind11::handle>(dtype))));
}

static auto date(const py::array &array) -> py::array_t<dateutils::Date> {
  auto frac = fractional_seconds_from_dtype(array.dtype());
  auto result =
      py::array_t<dateutils::Date>(py::array::ShapeContainer({array.size()}));
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<1>();

  {
    auto gil = py::gil_scoped_release();

    for (auto ix = 0; ix < array.size(); ++ix) {
      _result[ix] = dateutils::date_from_days(
          std::get<0>(frac.days_since_epoch(_array[ix])));
    }
  }
  return result;
}

static auto time(const py::array &array) -> py::array_t<dateutils::Time> {
  auto frac = fractional_seconds_from_dtype(array.dtype());
  auto result =
      py::array_t<dateutils::Time>(py::array::ShapeContainer({array.size()}));
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<1>();

  {
    auto gil = py::gil_scoped_release();

    for (auto ix = 0; ix < array.size(); ++ix) {
      _result[ix] = dateutils::time_from_seconds(
          std::get<1>(frac.days_since_epoch(_array[ix])));
    }
  }
  return result;
}

static auto isocalendar(const py::array &array)
    -> py::array_t<dateutils::ISOCalendar> {
  auto frac = fractional_seconds_from_dtype(array.dtype());
  auto result = py::array_t<dateutils::ISOCalendar>(
      py::array::ShapeContainer({array.size()}));
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<1>();

  {
    auto gil = py::gil_scoped_release();

    for (auto ix = 0; ix < array.size(); ++ix) {
      _result[ix] = dateutils::isocalendar(
          std::get<0>(frac.days_since_epoch(_array[ix])));
    }
  }
  return result;
}

static auto weekday(const py::array &array) -> py::array_t<unsigned> {
  auto frac = fractional_seconds_from_dtype(array.dtype());
  auto result =
      py::array_t<unsigned>(py::array::ShapeContainer({array.size()}));
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<1>();

  {
    auto gil = py::gil_scoped_release();

    for (auto ix = 0; ix < array.size(); ++ix) {
      _result[ix] =
          dateutils::weekday(std::get<0>(frac.days_since_epoch(_array[ix])));
    }
  }
  return result;
}

static auto timedelta_since_january(const py::array &array) -> py::array {
  auto frac = fractional_seconds_from_dtype(array.dtype());
  auto result =
      py::array(py::dtype(static_cast<std::string>(dateutils::DType(
                    dateutils::DType::kTimedelta64, frac.resolution()))),
                py::array::ShapeContainer({array.size()}), nullptr);
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<int64_t, 1>();

  {
    auto gil = py::gil_scoped_release();

    for (auto ix = 0; ix < array.size(); ++ix) {
      auto [days, seconds, fractional_part] = frac.days_since_epoch(_array[ix]);
      auto days_since_january =
          dateutils::days_since_january(dateutils::date_from_days(days));
      auto hms = dateutils::time_from_seconds(seconds);
      _result[ix] = (days_since_january * 86400LL + hms.hour * 3600LL +
                     hms.minute * 60LL + hms.second) *
                        frac.order_of_magnitude() +
                    fractional_part;
    }
  }
  return result;
}

static auto datetime(const py::array &array) -> py::array {
  auto frac = fractional_seconds_from_dtype(array.dtype());
  auto *buffer = new PyObject *[array.size()];
  auto _array = array.unchecked<int64_t, 1>();

  if (PyDateTimeAPI == nullptr) {
    PyDateTime_IMPORT;
  }

  for (auto ix = 0; ix < array.size(); ++ix) {
    auto [days, seconds, fractional_part] = frac.days_since_epoch(_array[ix]);
    auto date = dateutils::date_from_days(days);
    auto time = dateutils::time_from_seconds(seconds);

    buffer[ix] = PyDateTime_FromDateAndTime(
        date.year, date.month, date.day, time.hour, time.minute, time.second,
        static_cast<int>(frac.cast(fractional_part, dateutils::kMicrosecond)));
  }
  auto capsule = py::capsule(
      buffer, [](void *ptr) { delete[] static_cast<PyObject *>(ptr); });

  return py::array(py::dtype("object"),
                   pybind11::array::ShapeContainer({array.size()}), buffer,
                   capsule);
}

}  // namespace detail

void init_dateutils(py::module &m) {
  PYBIND11_NUMPY_DTYPE(dateutils::Date, year, month, day);
  PYBIND11_NUMPY_DTYPE(dateutils::Time, hour, minute, second);
  PYBIND11_NUMPY_DTYPE(dateutils::ISOCalendar, year, week, weekday);

  m.def("date", &detail::date, py::arg("array"), R"__doc__(
Return the date part of the dates.

Args:
    array: Numpy array of datetime64 to process.

Returns:
    A structured numpy array containing three fields: ``year``, ``month`` and
    ``day``.
)__doc__")
      .def("datetime", &detail::datetime, py::arg("array"), R"__doc__(
Return the data as an array of native Python datetime objects.

Args:
    array: Numpy array of datetime64 to process.

Returns:
    Object dtype array containing native Python datetime objects.
)__doc__")
      .def("timedelta_since_january", &detail::timedelta_since_january,
           py::arg("array"),
           R"__doc__(
Return the number the timedelta since the first January.

Args:
    array: Numpy array of datetime64 to process.

Returns:
    timedelta64 dtype array containing the time delta since the first January.
)__doc__")
      .def("isocalendar", &detail::isocalendar, py::arg("array"),
           R"__doc__(
Return the ISO calendar of dates.

Args:
    array: Numpy array of datetime64 to process.

Returns:
    A structured numpy array containing three fields: ``year``, ``week`` and
    ``weekday``.

.. seealso:: datetime.date.isocalendar.
)__doc__")
      .def("time", &detail::time, py::arg("array"), R"__doc__(
Return the time part of the dates.

Args:
    array: Numpy array of datetime64 to process.

Returns:
    A structured numpy array containing three fields: ``hour``, ``minute`` and
    ``second``.
)__doc__")
      .def("weekday", &detail::weekday, py::arg("array"), R"__doc__(
Return the weekday of the dates; Sunday is 0 ... Saturday is 6.

Args:
    array: Numpy array of datetime64 to process.

Returns:
    Int dtype array containing weekday of the dates.
)__doc__")
      // Intentionally undocumented: this function is used only for unit tests
      .def(
          "datetime64_to_str",
          [](const int64_t value,
             const std::string &clock_resolution) -> std::string {
            return pyinterp::dateutils::datetime64_to_string(
                value, dateutils::DType(clock_resolution));
          },
          py::arg("value"), py::arg("resolution"));
}
