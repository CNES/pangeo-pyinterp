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

  {
    auto gil = py::gil_scoped_release();

    for (auto ix = 0; ix < array.size(); ++ix) {
      _result[ix] = dateutils::year_month_day(frac.seconds(_array[ix]));
    }
  }
  return result;
}

static auto time(const py::array& array) -> py::array_t<dateutils::Time> {
  auto frac = dateutils::FractionalSeconds(array.dtype());
  auto result =
      py::array_t<dateutils::Time>(py::array::ShapeContainer({array.size()}));
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<1>();

  {
    auto gil = py::gil_scoped_release();

    for (auto ix = 0; ix < array.size(); ++ix) {
      _result[ix] = dateutils::hour_minute_second(frac.seconds(_array[ix]));
    }
  }
  return result;
}

static auto isocalendar(const py::array& array)
    -> py::array_t<dateutils::ISOCalendar> {
  auto frac = dateutils::FractionalSeconds(array.dtype());
  auto result = py::array_t<dateutils::ISOCalendar>(
      py::array::ShapeContainer({array.size()}));
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<1>();

  {
    auto gil = py::gil_scoped_release();

    for (auto ix = 0; ix < array.size(); ++ix) {
      _result[ix] = dateutils::isocalendar(frac.seconds(_array[ix]));
    }
  }
  return result;
}

static auto weekday(const py::array& array) -> py::array_t<unsigned> {
  auto frac = dateutils::FractionalSeconds(array.dtype());
  auto result =
      py::array_t<unsigned>(py::array::ShapeContainer({array.size()}));
  auto _array = array.unchecked<int64_t, 1>();
  auto _result = result.mutable_unchecked<1>();

  {
    auto gil = py::gil_scoped_release();

    for (auto ix = 0; ix < array.size(); ++ix) {
      _result[ix] = dateutils::weekday(frac.seconds(_array[ix]));
    }
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

  {
    auto gil = py::gil_scoped_release();

    for (auto ix = 0; ix < array.size(); ++ix) {
      _result[ix] = dateutils::days_since_january(
          dateutils::year_month_day(frac.seconds(_array[ix])));
    }
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

    buffer[ix] = PyDateTime_FromDateAndTime(date.year, date.month, date.day,
                                            time.hour, time.minute, time.second,
                                            static_cast<int>(msec));
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
  PYBIND11_NUMPY_DTYPE(dateutils::ISOCalendar, year, week, weekday);

  m.def("date", &detail::date, py::arg("array"), R"__doc__(
Return the date part of the dates.

Args:
    array (numpy.ndarray): Numpy array of datetime64 to process

Return:
    numpy.ndarray: A structured numpy array containing three fields: ``year``,
    ``month`` and ``day``.
)__doc__")
      .def("datetime", &detail::datetime, py::arg("array"), R"__doc__(
Return the data as an array of native Python datetime objects.

Args:
    array (numpy.ndarray): Numpy array of datetime64 to process

Return:
    numpy.ndarray: Object dtype array containing native Python datetime objects.
)__doc__")
      .def("days_since_january", &detail::days_since_january, py::arg("array"),
           R"__doc__(
Return the number of days since the first January.

Args:
    array (numpy.ndarray): Numpy array of datetime64 to process

Return:
    numpy.ndarray: integer dtype array containing the number of days since the
    first January.
)__doc__")
      .def("isocalendar", &detail::isocalendar, py::arg("array"),
           R"__doc__(
Return the ISO calendar of dates.

Args:
    array (numpy.ndarray): Numpy array of datetime64 to process

Return:
    numpy.ndarray: A structured numpy array containing three fields: ``year``,
    ``week`` and ``weekday``.

.. seealso:: datetime.date.isocalendar
)__doc__")
      .def("time", &detail::time, py::arg("array"), R"__doc__(
Return the time part of the dates.

Args:
    array (numpy.ndarray): Numpy array of datetime64 to process

Return:
    numpy.ndarray: A structured numpy array containing three fields: ``hour``,
    ``minute`` and ``second``.
)__doc__")
      .def("weekday", &detail::weekday, py::arg("array"), R"__doc__(
Return the weekday of the dates; Sunday is 0 ... Saturday is 6.

Args:
    array (numpy.ndarray): Numpy array of datetime64 to process

Return:
    numpy.ndarray: int dtype array containing weekday of the dates.
)__doc__");
}
