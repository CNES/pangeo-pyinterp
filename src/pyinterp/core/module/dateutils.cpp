// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/dateutils.hpp"

#include <datetime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace dateutils = pyinterp::dateutils;

namespace detail {

static auto date(const py::array& array) -> py::array_t<dateutils::Date> {
  auto frac = dateutils::FractionalSeconds(array.dtype());
  auto result =
      py::array_t<dateutils::Date>(py::array::ShapeContainer({array.size()}));
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<1>();

  for (auto ix = 0; ix < array.size(); ++ix) {
    _result[ix] = dateutils::year_month_day(frac.seconds(_array[ix]));
  }
  return result;
}

static auto time(const py::array& array) -> py::array_t<dateutils::Time> {
  auto frac = dateutils::FractionalSeconds(array.dtype());
  auto result =
      py::array_t<dateutils::Time>(py::array::ShapeContainer({array.size()}));
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<1>();

  for (auto ix = 0; ix < array.size(); ++ix) {
    _result[ix] = dateutils::hour_minute_second(frac.seconds(_array[ix]));
  }
  return result;
}

static auto isocalandar(const py::array& array)
    -> py::array_t<dateutils::ISOCalandar> {
  auto frac = dateutils::FractionalSeconds(array.dtype());
  auto result = py::array_t<dateutils::ISOCalandar>(
      py::array::ShapeContainer({array.size()}));
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<1>();

  for (auto ix = 0; ix < array.size(); ++ix) {
    _result[ix] = dateutils::isocalendar(frac.seconds(_array[ix]));
  }
  return result;
}

static auto weekday(const py::array& array) -> py::array_t<unsigned> {
  auto frac = dateutils::FractionalSeconds(array.dtype());
  auto result =
      py::array_t<unsigned>(py::array::ShapeContainer({array.size()}));
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<1>();

  for (auto ix = 0; ix < array.size(); ++ix) {
    _result[ix] = dateutils::weekday(frac.seconds(_array[ix]));
  }
  return result;
}

static auto days_since_january(const py::array& array)
    -> py::array_t<unsigned> {
  auto frac = dateutils::FractionalSeconds(array.dtype());
  auto result =
      py::array_t<unsigned>(py::array::ShapeContainer({array.size()}));
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<1>();

  for (auto ix = 0; ix < array.size(); ++ix) {
    _result[ix] = dateutils::days_since_january(
        dateutils::year_month_day(frac.seconds(_array[ix])));
  }
  return result;
}

static auto datetime(const py::array& array) -> py::array {
  auto frac = dateutils::FractionalSeconds(array.dtype());
  auto* buffer = new PyObject*[array.size()];
  auto _array = array.unchecked<int64_t, 1>();

  if (!PyDateTimeAPI) {
    PyDateTime_IMPORT;
  }

  for (auto ix = 0; ix < array.size(); ++ix) {
    auto epoch = frac.seconds(_array[ix]);
    auto date = dateutils::year_month_day(epoch);
    auto time = dateutils::hour_minute_second(epoch);
    auto msec = frac.microsecond(_array[ix]);

    buffer[ix] =
        PyDateTime_FromDateAndTime(date.year, date.month, date.day, time.hour,
                                   time.minute, time.second, msec);
  }
  auto capsule = py::capsule(
      buffer, [](void* ptr) { delete[] static_cast<PyObject*>(ptr); });

  return py::array(py::dtype("object"),
                   pybind11::array::ShapeContainer({array.size()}), buffer,
                   capsule);
}

}  // namespace detail

void init_dateutils(py::module& m) {
  PYBIND11_NUMPY_DTYPE(dateutils::Date, year, month, day);
  PYBIND11_NUMPY_DTYPE(dateutils::Time, hour, minute, second);
  PYBIND11_NUMPY_DTYPE(dateutils::ISOCalandar, year, week_number, weekday);

  m.def("date", &detail::date, py::arg("array"))
      .def("datetime", &detail::datetime, py::arg("array"))
      .def("days_since_january", &detail::days_since_january, py::arg("array"))
      .def("isocalandar", &detail::isocalandar, py::arg("array"))
      .def("time", &detail::time, py::arg("array"))
      .def("weekday", &detail::weekday, py::arg("array"));
}
