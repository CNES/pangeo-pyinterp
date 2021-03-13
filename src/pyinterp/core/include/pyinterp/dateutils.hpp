// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

// Algorithms are extracted from the documentation of Howard Hinnant
// cf. http://howardhinnant.github.io/date_algorithms.html

namespace pyinterp {
namespace dateutils {
namespace detail {

constexpr int ISO_WEEK_START_WDAY = 1;  // Monday
constexpr int ISO_WEEK1_WDAY = 4;
constexpr int YDAY_MINIMUM = -366;
constexpr std::array<int, 13> DAYS_IN_MONTH({-1, 31, 28, 31, 30, 31, 30, 31, 31,
                                             30, 31, 30, 31});

// The number of days from the first day of the first ISO week of this year to
// the year day *yday* with week day *wday*.  ISO weeks start on Monday; the
// first ISO week has the year's first Thursday. *yday* may be as small as
// *YDAY_MINIMUM*.
auto iso_week_days(int yday, int wday) -> int {
  const int big_enough_multiple_of_7 = (-YDAY_MINIMUM / 7 + 2) * 7;
  return yday - (yday - wday + ISO_WEEK1_WDAY + big_enough_multiple_of_7) % 7 +
         ISO_WEEK1_WDAY - ISO_WEEK_START_WDAY;
}

// True if leap year, else false.
inline auto is_leap_year(const int year) -> int {
  return (year & 3) == 0 && ((year % 25) != 0 || (year & 15) == 0);  // NOLINT
}

// Number of days in that month in that year
inline auto days_in_month(const int year, const int month) -> int {
  if (month == 2 && is_leap_year(year) == 1) {
    return 29;
  }
  return DAYS_IN_MONTH[month];  // NOLINT
}

}  // namespace detail

/// Represents a date
struct Date {
  int year;
  unsigned month;
  unsigned day;
};

/// Represents a time
struct Time {
  unsigned hour;
  unsigned minute;
  unsigned second;
};

/// Represents an ISO calandar
struct ISOCalendar {
  int year;
  unsigned week;
  unsigned weekday;
};

/// Handle fractionnal seconds from numpy storage
class FractionalSeconds {
 public:
  /// Default constructor
  FractionalSeconds(const pybind11::dtype& dtype) {
    auto type_num =
        pybind11::detail::array_descriptor_proxy(dtype.ptr())->type_num;
    if (type_num != 21 /* NPY_DATETIME */) {
      throw std::invalid_argument(
          "array must be a numpy array of datetime64 items");
    }
    auto units =
        std::string(pybind11::str(static_cast<pybind11::handle>(dtype)));
    if (units == "datetime64[as]") {
      scale_ = 1'000'000'000'000'000'000;
    } else if (units == "datetime64[fs]") {
      scale_ = 1'000'000'000'000'000;
    } else if (units == "datetime64[ps]") {
      scale_ = 1'000'000'000'000;
    } else if (units == "datetime64[ns]") {
      scale_ = 1'000'000'000;
    } else if (units == "datetime64[us]") {
      scale_ = 1'000'000;
    } else if (units == "datetime64[ms]") {
      scale_ = 1'000;
    } else if (units == "datetime64[s]") {
      scale_ = 1;
    } else {
      throw std::invalid_argument(
          "array has wrong datetime unit, expected datetime64[as], "
          "datetime64[fs], datetime64[ns], datetime64[us], datetime64[ms] or "
          "datetime64[s] got " +
          units);
    }
  }

  /// Gets the numpy units
  inline auto units() const -> std::string {
    switch (scale_) {
      case 1'000'000'000'000'000'000:
        return "as";
      case 1'000'000'000'000'000:
        return "fs";
      case 1'000'000'000'000:
        return "ps";
      case 1'000'000'000:
        return "ns";
      case 1'000'000:
        return "us";
      case 1'000:
        return "ms";
      default:
        return "s";
    }
  }

  /// Gets the number of seconds elpased since 1970
  inline auto seconds(const int64_t datetime64) const noexcept -> int64_t {
    return datetime64 / scale_;
  }

  /// Gets the fractional part of the date
  inline auto fractional(const int64_t datetime64) const noexcept -> int64_t {
    return datetime64 % scale_;
  }

  /// Get the number of microseconds contained in the date.
  inline auto microsecond(const int64_t datetime64) const noexcept -> int64_t {
    auto frac = fractional(datetime64);
    return scale_ <= 1'000'000 ? (1'000'000 / scale_) * frac
                               : frac / (scale_ / 1'000'000);
  }

  /// Get the numpy scale
  inline auto scale() const noexcept -> int64_t { return scale_; }

 private:
  int64_t scale_;
};

/// Gets year, month, day in civil calendar
auto year_month_day(const int64_t epoch) noexcept -> Date {
  const auto z =
      epoch / 86400LL + 719468LL - static_cast<int64_t>((epoch % 86400LL) < 0);
  const auto era = static_cast<int>((z >= 0 ? z : z - 146096LL) / 146097LL);
  const auto doe = static_cast<unsigned>(z - era * 146097LL);  // [0, 146096]
  const auto yoe =
      (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;  // [0, 399]
  const auto doy = doe - (365 * yoe + yoe / 4 - yoe / 100);   // [0, 365]
  const auto mp = (5 * doy + 2) / 153;                        // [0, 11]
  const auto d = doy - (153 * mp + 2) / 5 + 1;                // [1, 31]
  const auto m = mp < 10 ? mp + 3 : mp - 9;                   // [1, 12]

  return {(static_cast<int>(yoe) + era * 400) + static_cast<int>(m <= 2), m, d};
}

/// Gets the number of hours, minutes and seconds elapsed in the day
auto hour_minute_second(const int64_t epoch) noexcept -> Time {
  auto seconds_within_day = epoch % 86400;
  if (seconds_within_day < 0) {
    seconds_within_day += 86400;
  }
  const auto seconds_within_hour = seconds_within_day % 3600;

  return {static_cast<unsigned>(seconds_within_day / 3600),
          static_cast<unsigned>(seconds_within_hour / 60),
          static_cast<unsigned>(seconds_within_hour % 60)};
}

/// Gets the number of days since the first January
auto days_since_january(const Date& date) -> unsigned {
  unsigned result = date.day - 1;

  if (date.month > 2) {
    result += detail::is_leap_year(date.year);
  }

  for (size_t ix = 1; ix < date.month; ++ix) {
    result += detail::DAYS_IN_MONTH[ix];
  }

  return result;
}

/// Gets the day of the week; Sunday is 0 ... Saturday is 6
inline auto weekday(const int64_t epoch) noexcept -> unsigned {
  const auto days = epoch / 86400;
  return static_cast<unsigned>(days >= -4 ? (days + 4) % 7
                                          : (days + 5) % 7 + 6);
}

/// Return the ISO calandar
///
/// The first ISO week of the year is the (Mon-Sun) week
/// containing the year's first Thursday; everything else derives
/// from that.
///
///  The first week is 1; Monday is 1 ... Sunday is 7.
auto isocalendar(const int64_t epoch) -> ISOCalendar {
  auto date = year_month_day(epoch);
  auto yday = days_since_january(date);
  auto wday = weekday(epoch);
  auto days = detail::iso_week_days(yday, wday);

  // This ISO week belongs to the previous year ?
  if (days < 0) {
    date.year--;
    days = detail::iso_week_days(yday + (365 + detail::is_leap_year(date.year)),
                                 wday);
  } else {
    int week_days = detail::iso_week_days(
        yday - (365 + detail::is_leap_year(date.year)), wday);

    // This ISO week belongs to the next year ?
    if (0 <= week_days) {
      date.year++;
      days = week_days;
    }
  }
  return {date.year, static_cast<unsigned>(days / 7 + 1),
          (wday - 1 + 7) % 7 + 1};
}

}  // namespace dateutils
}  // namespace pyinterp