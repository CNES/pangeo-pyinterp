// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

#include "pyinterp/detail/math.hpp"

// Algorithms are extracted from the documentation of Howard Hinnant
// cf. http://howardhinnant.github.io/date_algorithms.html

namespace pyinterp::dateutils {
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
constexpr auto iso_week_days(int yday, int wday) -> int {
  constexpr int big_enough_multiple_of_7 = (-YDAY_MINIMUM / 7 + 2) * 7;
  return yday - (yday - wday + ISO_WEEK1_WDAY + big_enough_multiple_of_7) % 7 +
         ISO_WEEK1_WDAY - ISO_WEEK_START_WDAY;
}

// True if leap year, else false.
constexpr auto is_leap_year(const int year) -> int {
  return (year & 3) == 0 && ((year % 25) != 0 || (year & 15) == 0);  // NOLINT
}

// Number of days in that month in that year
constexpr auto days_in_month(const int year, const int month) -> int {
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
  explicit FractionalSeconds(const pybind11::dtype& dtype) {
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
  [[nodiscard]] inline auto units() const -> std::string {
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

  /// Gets the number of seconds elpased since 1970 and the fractional part
  [[nodiscard]] constexpr auto epoch(const int64_t datetime64) const noexcept
      -> std::tuple<int64_t, int64_t> {
    auto sec = seconds(datetime64);
    auto frac = fractional(datetime64);
    if (frac < 0) {
      frac += scale_;
      sec -= 1;
    }
    return std::make_tuple(sec, frac);
  }

  [[nodiscard]] constexpr auto fractional_part(
      const int64_t frac, const int64_t scale) const noexcept -> int64_t {
    return scale_ <= scale ? (scale / scale_) * frac : frac / (scale_ / scale);
  }

  /// Gets the number of days, seconds and the fractional part elapsed since
  /// 1970
  [[nodiscard]] constexpr auto days_since_epoch(const int64_t datetime64)
      const noexcept -> std::tuple<int64_t, int64_t, int64_t> {
    auto [seconds, fractional] = epoch(datetime64);
    auto days = seconds / 86400LL;
    if (seconds % 86400LL < 0) {
      --days;
    }
    return std::tuple(days, seconds, fractional);
  }

  /// Gets the numpy scale
  [[nodiscard]] constexpr auto scale() const noexcept -> int64_t {
    return scale_;
  }

  /// Gets the maximum number of digits for the fractional part
  [[nodiscard]] inline auto ndigits() const noexcept -> int {
    return static_cast<int>(std::log10(scale_));
  }

 private:
  int64_t scale_;

  /// Gets the number of seconds elpased since 1970
  [[nodiscard]] constexpr auto seconds(const int64_t datetime64) const noexcept
      -> int64_t {
    return datetime64 / scale_;
  }

  /// Gets the fractional part of the date
  [[nodiscard]] constexpr auto fractional(
      const int64_t datetime64) const noexcept -> int64_t {
    return datetime64 % scale_;
  }
};

/// Gets year, month, day in civil calendar
constexpr auto date_from_days_since_epoch(int64_t days_since_epoch) noexcept
    -> Date {
  days_since_epoch += 719468LL;
  // era : 400 year period
  const auto era = static_cast<int>(
      (days_since_epoch >= 0 ? days_since_epoch : days_since_epoch - 146096LL) /
      146097LL);
  // day of era
  const auto doe =
      static_cast<unsigned>(days_since_epoch - era * 146097LL);  // [0, 146096]
  // year of era
  const auto yoe =
      (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;  // [0, 399]
  // day of year
  const auto doy = doe - (365 * yoe + yoe / 4 - yoe / 100);  // [0, 365]
  // month number (March=0, April=1, May=2, ..., January=10, February=11)
  const auto mon = (5 * doy + 2) / 153;             // [0, 11]
  const auto day = doy - (153 * mon + 2) / 5 + 1;   // [1, 31]
  const auto month = mon < 10 ? mon + 3 : mon - 9;  // [1, 12]

  return {(static_cast<int>(yoe) + era * 400) + static_cast<int>(month <= 2),
          month, day};
}

/// Gets year, month, day in civil calendar
constexpr auto date_from_epoch(const int64_t epoch) noexcept -> Date {
  // number of days since 1970-01-01
  auto days_since_epoch =
      epoch / 86400LL - static_cast<int64_t>((epoch % 86400LL) < 0);
  return date_from_days_since_epoch(days_since_epoch);
}

/// Gets the number of hours, minutes and seconds elapsed in the day
constexpr auto time_from_epoch(const int64_t epoch) noexcept -> Time {
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
constexpr auto days_since_january(const Date& date) -> unsigned {
  unsigned result = date.day - 1;

  if (date.month > 2) {
    result += detail::is_leap_year(date.year);
  }

  for (size_t ix = 1; ix < date.month; ++ix) {
    result += detail::DAYS_IN_MONTH[ix];
  }

  return result;
}

/// Gets thweeke day of the week; Sunday is 0 ... Saturday is 6
constexpr auto weekday(const int64_t days) noexcept -> unsigned {
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
constexpr auto isocalendar(const int64_t days_since_epoch) -> ISOCalendar {
  auto date = date_from_days_since_epoch(days_since_epoch);
  auto yday = days_since_january(date);
  auto wday = weekday(days_since_epoch);
  auto days =
      detail::iso_week_days(static_cast<int>(yday), static_cast<int>(wday));

  // This ISO week belongs to the previous year ?
  if (days < 0) {
    date.year--;
    days = detail::iso_week_days(
        static_cast<int>(yday) + (365 + detail::is_leap_year(date.year)),
        static_cast<int>(wday));
  } else {
    int week_days = detail::iso_week_days(
        static_cast<int>(yday) - (365 + detail::is_leap_year(date.year)),
        static_cast<int>(wday));

    // This ISO week belongs to the next year ?
    if (0 <= week_days) {
      date.year++;
      days = week_days;
    }
  }
  return {date.year, static_cast<unsigned>(days / 7 + 1),
          (wday - 1 + 7) % 7 + 1};
}

/// Get the date from the number of years since 1970-01-01
constexpr auto date_from_years_since_epoch(const int64_t years) -> Date {
  return {static_cast<int>(1970 + years), 1, 1};
}

/// Get the date from the number of months since 1970-01-01
constexpr auto date_from_months_since_epoch(const int64_t value) -> Date {
  auto years = value / 12;
  auto months = value % 12;
  if (months != 0 && value < 0) {
    --years;
    months += 12;
  }
  return {static_cast<int>(1970 + years), static_cast<unsigned>(months + 1), 1};
}

/// Get the date from the number of weeks since 1970-01-01
constexpr auto date_from_weeks_since_epoch(const int64_t weeks) -> Date {
  return date_from_days_since_epoch(weeks * 7);
}

/// Get the date from the number of hours since 1970-01-01
constexpr auto date_from_hours_since_epoch(const int64_t value)
    -> std::tuple<Date, Time> {
  auto days = value / 24;
  auto hours = value % 24;
  if (hours != 0 && value < 0) {
    --days;
    hours += 24;
  }
  return std::make_tuple(date_from_days_since_epoch(days),
                         Time{static_cast<unsigned>(hours), 0, 0});
}

/// Get the date from the number of minutes since 1970-01-01
constexpr auto date_from_minutes_since_epoch(const int64_t value)
    -> std::tuple<Date, Time> {
  auto days = value / 1440;
  auto minutes = value % 1440;
  if (minutes != 0 && value < 0) {
    --days;
    minutes += 1440;
  }
  return std::make_tuple(date_from_days_since_epoch(days),
                         time_from_epoch(minutes * 60));
}

/// Return a string representation of the numpy datetime64
[[nodiscard]] inline auto datetime64_to_str(const int64_t value,
                                            const std::string& resolution)
    -> std::string {
  auto date = Date{};
  auto time = Time{};
  auto ss = std::stringstream{};

  // Value is encoded as years elapsed since 1970.
  if (resolution == "Y") {
    date = date_from_years_since_epoch(value);
    ss << date.year;

    // Value is encoded as months elapsed since 1970.
  } else if (resolution == "M") {
    date = date_from_months_since_epoch(value);
    ss << date.year << "-" << std::setfill('0') << std::setw(2)
       << std::to_string(date.month);

    // Value is encoded as weeks elapsed since 1970.
  } else if (resolution == "W") {
    date = date_from_weeks_since_epoch(value);
    ss << date.year << "-" << std::setfill('0') << std::setw(2) << date.month
       << "-" << std::setfill('0') << std::setw(2) << date.day;

    // Value is encoded as days elapsed since 1970.
  } else if (resolution == "D") {
    date = date_from_days_since_epoch(value);
    ss << date.year << "-" << std::setfill('0') << std::setw(2) << date.month
       << "-" << std::setfill('0') << std::setw(2) << date.day;

    // Value is encoded as hours elapsed since 1970.
  } else if (resolution == "h") {
    std::tie(date, time) = date_from_hours_since_epoch(value);
    ss << date.year << "-" << std::setfill('0') << std::setw(2) << date.month
       << "-" << std::setfill('0') << std::setw(2) << date.day << "T"
       << std::setfill('0') << std::setw(2) << time.hour;

    // Value is encoded as minutes elapsed since 1970.
  } else if (resolution == "m") {
    std::tie(date, time) = date_from_minutes_since_epoch(value);
    ss << date.year << "-" << std::setfill('0') << std::setw(2) << date.month
       << "-" << std::setfill('0') << std::setw(2) << date.day << "T"
       << std::setfill('0') << std::setw(2) << time.hour << ":"
       << std::setfill('0') << std::setw(2) << time.minute;
  }

  // If resolution is one of the above, return the string.
  if (ss.tellp() != 0) {
    return ss.str();
  }

  // Value is encoded as fractional seconds elapsed since 1970 (the constructor
  // throws an exception if the resolution is invalid).
  auto frac =
      FractionalSeconds{pybind11::dtype("datetime64[" + resolution + "]")};
  auto [sec, fractional_part] = frac.epoch(value);

  date = date_from_epoch(sec);
  time = time_from_epoch(sec);

  // Write the date and time.
  ss << date.year << "-" << std::setfill('0') << std::setw(2) << date.month
     << "-" << std::setfill('0') << std::setw(2) << date.day << "T"
     << std::setfill('0') << std::setw(2) << time.hour << ":"
     << std::setfill('0') << std::setw(2) << time.minute << ":"
     << std::setfill('0') << std::setw(2) << time.second;

  int ndigits = frac.ndigits();
  if (ndigits > 0) {
    ss << "." << std::setfill('0') << std::setw(ndigits)
       << std::to_string(fractional_part);
  }
  return ss.str();
}

}  // namespace pyinterp::dateutils