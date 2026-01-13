// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/dateutils.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <ranges>

#include "pyinterp/dateutils.hpp"
#include "pyinterp/pybind/numpy.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::pybind {
namespace detail {

struct ExtractDatetime64 {
  dateutils::DType dtype;
  dateutils::FractionalSeconds frac;
  Vector<int64_t> mapped_integer_values;

  [[nodiscard]] constexpr auto size() const -> int64_t {
    return mapped_integer_values.size();
  }
};

auto extract_datetime64(const nb::object &array) -> ExtractDatetime64 {
  auto dtype = retrieve_dtype("points", array);
  if (dtype.datetype() != dateutils::DType::DateType::kDatetime64) {
    throw std::invalid_argument(
        "array must be a numpy.datetime64 array for date extraction, got " +
        static_cast<std::string>(dtype));
  }
  auto mapped_integer_values = numpy_to_vector(array);
  auto frac = dateutils::FractionalSeconds(dtype);
  return {.dtype = dtype,
          .frac = frac,
          .mapped_integer_values = mapped_integer_values};
}

}  // namespace detail

inline auto date(const nb::object &array)
    -> std::tuple<Vector<int32_t>, Vector<uint8_t>, Vector<uint8_t>> {
  auto datetime64 = detail::extract_datetime64(array);
  auto size = datetime64.size();
  auto years = Vector<int32_t>(size);
  auto months = Vector<uint8_t>(size);
  auto days = Vector<uint8_t>(size);
  {
    nb::gil_scoped_release release;

    for (auto [epoch, year, month, day] : std::views::zip(
             datetime64.mapped_integer_values, years, months, days)) {
      auto date = dateutils::date_from_days(
          std::get<0>(datetime64.frac.days_since_epoch(epoch)));
      year = static_cast<int32_t>(date.year);
      month = static_cast<uint8_t>(date.month);
      day = static_cast<uint8_t>(date.day);
    }
  }
  return {years, months, days};
}

inline auto time(const nb::object &array)
    -> std::tuple<Vector<uint8_t>, Vector<uint8_t>, Vector<uint8_t>> {
  auto datetime64 = detail::extract_datetime64(array);
  auto size = datetime64.size();
  auto hours = Vector<uint8_t>(size);
  auto minutes = Vector<uint8_t>(size);
  auto seconds = Vector<uint8_t>(size);
  {
    nb::gil_scoped_release release;

    for (auto [epoch, hour, minute, second] : std::views::zip(
             datetime64.mapped_integer_values, hours, minutes, seconds)) {
      auto time = dateutils::time_from_seconds(
          std::get<1>(datetime64.frac.days_since_epoch(epoch)));
      hour = static_cast<uint8_t>(time.hour);
      minute = static_cast<uint8_t>(time.minute);
      second = static_cast<uint8_t>(time.second);
    }
  }
  return {hours, minutes, seconds};
}

inline auto isocalendar(const nb::object &array)
    -> std::tuple<Vector<int32_t>, Vector<uint8_t>, Vector<uint8_t>> {
  auto datetime64 = detail::extract_datetime64(array);
  auto size = datetime64.size();
  auto years = Vector<int32_t>(size);
  auto weeks = Vector<uint8_t>(size);
  auto weekdays = Vector<uint8_t>(size);
  {
    nb::gil_scoped_release release;

    for (auto [epoch, year, week, weekday] : std::views::zip(
             datetime64.mapped_integer_values, years, weeks, weekdays)) {
      auto iso_calendar = dateutils::isocalendar(
          std::get<0>(datetime64.frac.days_since_epoch(epoch)));
      year = static_cast<int32_t>(iso_calendar.year);
      week = static_cast<uint8_t>(iso_calendar.week);
      weekday = static_cast<uint8_t>(iso_calendar.weekday);
    }
  }
  return {years, weeks, weekdays};
}

auto weekday(const nb::object &array) -> Vector<uint8_t> {
  auto datetime64 = detail::extract_datetime64(array);
  auto weekdays = Vector<uint8_t>(datetime64.size());
  {
    nb::gil_scoped_release release;

    for (auto [epoch, weekday] :
         std::views::zip(datetime64.mapped_integer_values, weekdays)) {
      weekday = static_cast<uint8_t>(dateutils::weekday(
          std::get<0>(datetime64.frac.days_since_epoch(epoch))));
    }
  }
  return weekdays;
}

auto timedelta_since_january(const nb::object &array) -> nb::object {
  auto datetime64 = detail::extract_datetime64(array);
  auto results = Vector<int64_t>(datetime64.size());
  {
    nb::gil_scoped_release release;

    for (auto [epoch, result] :
         std::views::zip(datetime64.mapped_integer_values, results)) {
      auto [days, seconds, fractional_part] =
          datetime64.frac.days_since_epoch(epoch);
      auto days_since_january =
          dateutils::days_since_january(dateutils::date_from_days(days));
      auto hms = dateutils::time_from_seconds(seconds);
      result = (days_since_january * 86400LL + hms.hour * 3600LL +
                hms.minute * 60LL + hms.second) *
                   datetime64.frac.order_of_magnitude() +
               fractional_part;
    }
  }
  return vector_to_numpy(std::move(results), datetime64.dtype.as_timedelta64());
}

constexpr auto kDateDoc = R"(
Extract the date part (year, month, day) from a numpy datetime64 array.

Args:
    array: Input numpy datetime64 array.
Returns:
    A tuple of three numpy arrays (years, months, days).
)";

constexpr auto kTimeDoc = R"(
Extract the time part (hour, minute, second) from a numpy datetime64 array.

Args:
    array: Input numpy datetime64 array.
Returns:
    A tuple of three numpy arrays (hours, minutes, seconds).
)";

constexpr auto kIsocalendarDoc = R"(
Extract the ISO calendar date (year, week number, weekday) from a numpy
datetime64 array.

The ISO year consists of 52 or 53 full weeks, and where a week starts on a
Monday and ends on a Sunday. The first week of an ISO year is the first
(Gregorian) calendar week of a year containing a Thursday. This is called week
number 1, and the ISO year of that Thursday is the same as its Gregorian year.

Args:
    array: Input numpy datetime64 array.
Returns:
    A tuple of three numpy arrays (ISO years, ISO weeks, ISO weekdays).
See Also:
    https://docs.python.org/3/library/datetime.html#datetime.date.isocalendar
)";

constexpr auto kWeekdayDoc = R"(
Return the day of the week as an integer, where Monday is 0 and Sunday is 6.

Args:
    array: Input numpy datetime64 array.
Returns:
    A numpy array of uint8 representing the day of the week.
)";

constexpr auto kTimedeltaSinceJanuaryDoc = R"(
Return the timedelta since the first day of the year.

Args:
    array: Input numpy datetime64 array.
Returns:
    A numpy timedelta64 array representing the time elapsed since the first
    day of the year.
)";

auto init_dateutils(nanobind::module_ &m) -> void {
  auto submodule = m.def_submodule("dateutils", "Date and time utilities");
  submodule.def("date", &date, kDateDoc, "array"_a);
  submodule.def("time", &time, kTimeDoc, "array"_a);
  submodule.def("isocalendar", &isocalendar, kIsocalendarDoc, "array"_a);
  submodule.def("weekday", &weekday, kWeekdayDoc, "array"_a);
  submodule.def("timedelta_since_january", &timedelta_since_january,
                kTimedeltaSinceJanuaryDoc, "array"_a);
}

}  // namespace pyinterp::pybind
