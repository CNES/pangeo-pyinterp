// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>

namespace pyinterp::dateutils {

constexpr int64_t kEpoch = 1970;

constexpr int64_t kDaysInWeek = 7;
constexpr int64_t kHoursInDay = 24;
constexpr int64_t kMinutesInHour = 60;
constexpr int64_t kMinutesInDay = kMinutesInHour * kHoursInDay;
constexpr int64_t kMonthsInYear = 12;
constexpr int64_t kWeeksInYear = 52;

constexpr int64_t kSecondsInMinute = 60;
constexpr int64_t kSecondsInHour = kSecondsInMinute * kMinutesInHour;
constexpr int64_t kSecondsInDay = kSecondsInHour * kHoursInDay;

constexpr int64_t kSecond = 1;
constexpr int64_t kMillisecond = 1'000;
constexpr int64_t kMicrosecond = 1'000'000;
constexpr int64_t kNanosecond = 1'000'000'000;
constexpr int64_t kPicoSecond = 1'000'000'000'000;
constexpr int64_t kFemtoSecond = 1'000'000'000'000'000;
constexpr int64_t kAttosecond = 1'000'000'000'000'000'000;

constexpr int kIsoWeekStartWDay = 1;  // Monday
constexpr int kIsoWeekFirstWDay = 4;

constexpr int kYDayMinimum = -366;

constexpr std::array<int, 13> kDaysInMonth({-1, 31, 28, 31, 30, 31, 30, 31, 31,
                                            30, 31, 30, 31});

/// Handles numpy encoded dates.
class DType {
 public:
  /// Date type
  enum DateType {
    kDatetime64,
    kTimedelta64,
  };

  /// Clock resolution
  enum Resolution {
    kYear = 0,
    kMonth = 1,
    kWeek = 2,
    kDay = 3,
    kHour = 4,
    kMinute = 5,
    kSecond = 6,
    kMillisecond = 7,
    kMicrosecond = 8,
    kNanosecond = 9,
    kPicosecond = 10,
    kFemtosecond = 11,
    kAttosecond = 12,
  };

  /// Default constructor
  explicit inline DType(const std::string &dtype) {
    auto match = std::smatch{};
    if (!std::regex_search(dtype, match, pattern_)) {
      throw std::invalid_argument("unknown numpy date type: " + dtype);
    }

    datetype_ = match[1] == "datetime64" ? kDatetime64 : kTimedelta64;
    resolution_ = DType::parse_unit(match[2]);
  }

  /// Build a DType from a date type and a resolution
  DType(const DateType datetype, const Resolution resolution)
      : datetype_{datetype}, resolution_{resolution} {}

  /// Get the clock resolution name
  [[nodiscard]] constexpr auto clock_name() const -> const char * {
    switch (resolution_) {
      case kYear:
        return "year";
      case kMonth:
        return "month";
      case kWeek:
        return "week";
      case kDay:
        return "day";
      case kHour:
        return "hour";
      case kMinute:
        return "minute";
      case kSecond:
        return "second";
      case kMillisecond:
        return "millisecond";
      case kMicrosecond:
        return "microsecond";
      case kNanosecond:
        return "nanosecond";
      case kPicosecond:
        return "picosecond";
      case kFemtosecond:
        return "femtosecond";
      case kAttosecond:
      default:
        return "attosecond";
    }
  }

  /// Get the clock unit
  [[nodiscard]] constexpr auto unit() const -> const char * {
    switch (resolution_) {
      case kYear:
        return "Y";
      case kMonth:
        return "M";
      case kWeek:
        return "W";
      case kDay:
        return "D";
      case kHour:
        return "h";
      case kMinute:
        return "m";
      case kSecond:
        return "s";
      case kMillisecond:
        return "ms";
      case kMicrosecond:
        return "us";
      case kNanosecond:
        return "ns";
      case kPicosecond:
        return "ps";
      case kFemtosecond:
        return "fs";
      case kAttosecond:
      default:
        return "as";
    }
  }

  /// Get the clock resolution handled by this instance
  [[nodiscard]] constexpr auto resolution() const -> Resolution {
    return resolution_;
  }

  /// Get the date type handled by this instance
  [[nodiscard]] constexpr auto datetype() const -> DateType {
    return datetype_;
  }

  /// Get the date type name
  [[nodiscard]] auto constexpr datetype_name() const -> const char * {
    switch (datetype_) {
      case kDatetime64:
        return "datetime64";
      // case kTimedelta64:
      default:
        return "timedelta64";
    }
  }

  /// Return the string representation of the numpy data type handled by this
  /// instance.
  explicit operator std::string() const {
    std::stringstream ss;
    ss << datetype_name() << "[" << unit() << "]";
    return ss.str();
  }

  /// Get the order of magnitude of the resolution between seconds and the
  /// clock resolution
  [[nodiscard]] constexpr auto order_of_magnitude() const -> int64_t {
    switch (resolution_) {
      case kSecond:
        return dateutils::kSecond;
      case kMillisecond:
        return dateutils::kMillisecond;
      case kMicrosecond:
        return dateutils::kMicrosecond;
      case kNanosecond:
        return dateutils::kNanosecond;
      case kPicosecond:
        return dateutils::kPicoSecond;
      case kFemtosecond:
        return dateutils::kFemtoSecond;
      case kAttosecond:
        return dateutils::kAttosecond;
      default:
        throw std::invalid_argument("The date resolution must be >= second");
    }
  }

  /// Compare two instances
  constexpr auto operator<(const DType &other) const -> bool {
    return compare(other) < 0;
  }

  /// Return true if the two instances are different.
  constexpr auto operator!=(const DType &other) const -> bool {
    return datetype_ != other.datetype_ || resolution_ != other.resolution_;
  }

  /// Return true if the two instances are equal.
  constexpr auto operator==(const DType &other) const -> bool {
    return datetype_ == other.datetype_ && resolution_ == other.resolution_;
  }

  /// Transform this instance into a new instance handling a timedelta64 type
  /// with the same resolution.
  [[nodiscard]] inline auto as_timedelta64() const -> DType {
    return {kTimedelta64, resolution_};
  }

 private:
  static const std::regex pattern_;
  DateType datetype_;
  Resolution resolution_;

  /// Compare two resolutions
  /// @return 0 if lhs == rhs, -1 if lhs < rhs, 1 if lhs > rhs
  [[nodiscard]] constexpr auto compare(const DType &rhs) const -> int {
    return static_cast<int>(resolution_) - static_cast<int>(rhs.resolution_);
  }

  /// Parse the resolution unit from a string
  [[nodiscard]] static inline auto parse_unit(const std::string &unit)
      -> Resolution {
    if (unit == "Y") {
      return kYear;
    }
    if (unit == "M") {
      return kMonth;
    }
    if (unit == "W") {
      return kWeek;
    }
    if (unit == "D") {
      return kDay;
    }
    if (unit == "h") {
      return kHour;
    }
    if (unit == "m") {
      return kMinute;
    }
    if (unit == "s") {
      return kSecond;
    }
    if (unit == "ms") {
      return kMillisecond;
    }
    if (unit == "us") {
      return kMicrosecond;
    }
    if (unit == "ns") {
      return kNanosecond;
    }
    if (unit == "ps") {
      return kPicosecond;
    }
    if (unit == "fs") {
      return kFemtosecond;
    }
    if (unit == "as") {
      return kAttosecond;
    }
    throw std::invalid_argument("invalid date unit: " + unit);
  }
};

/// Handle a date encoded in a 64-bit integer for a given clock resolution (
/// Clock resolution must be in range kSecond to kAttosecond)
class FractionalSeconds {
 public:
  /// Default constructor
  explicit FractionalSeconds(const DType &dtype)
      : order_of_magnitude_(dtype.order_of_magnitude()) {}

  /// Constructor from a numpy data type encoded in a string
  explicit FractionalSeconds(const std::string &dtype)
      : order_of_magnitude_(DType(dtype).order_of_magnitude()) {}

  /// Get the number of days, seconds and the fractional part elapsed since
  /// 1970
  [[nodiscard]] constexpr auto days_since_epoch(const int64_t datetime64)
      const noexcept -> std::tuple<int64_t, int64_t, int64_t> {
    auto [seconds, fractional] = epoch(datetime64);
    auto days = seconds / kSecondsInDay;
    if (seconds % kSecondsInDay < 0) {
      --days;
    }
    return std::make_tuple(days, seconds, fractional);
  }

  /// Get the maximum number of digits for the fractional part
  [[nodiscard]] inline auto ndigits() const noexcept -> int {
    return static_cast<int>(std::log10(order_of_magnitude_));
  }

  [[nodiscard]] constexpr auto cast(const int64_t frac,
                                    const int64_t scale) const noexcept
      -> int64_t {
    return order_of_magnitude_ <= scale ? (scale / order_of_magnitude_) * frac
                                        : frac / (order_of_magnitude_ / scale);
  }

  [[nodiscard]] constexpr auto order_of_magnitude() const noexcept -> int64_t {
    return order_of_magnitude_;
  }

  /// Get the numpy units
  [[nodiscard]] constexpr auto resolution() const -> DType::Resolution {
    switch (order_of_magnitude_) {
      case kAttosecond:
        return DType::kAttosecond;
      case kFemtoSecond:
        return DType::kFemtosecond;
      case kPicoSecond:
        return DType::kPicosecond;
      case kNanosecond:
        return DType::kNanosecond;
      case kMicrosecond:
        return DType::kMicrosecond;
      case kMillisecond:
        return DType::kMillisecond;
      default:
        return DType::kSecond;
    }
  }

 private:
  int64_t order_of_magnitude_;

  /// Get the number of seconds elapsed since 1970
  [[nodiscard]] constexpr auto seconds(const int64_t datetime64) const noexcept
      -> int64_t {
    return datetime64 / order_of_magnitude_;
  }

  /// Get the fractional part of the date
  [[nodiscard]] constexpr auto fractional(
      const int64_t datetime64) const noexcept -> int64_t {
    return datetime64 % order_of_magnitude_;
  }

  /// Get the number of seconds elapsed since 1970 and the fractional part
  [[nodiscard]] constexpr auto epoch(const int64_t datetime64) const noexcept
      -> std::tuple<int64_t, int64_t> {
    auto sec = seconds(datetime64);
    auto frac = fractional(datetime64);
    if (frac < 0) {
      frac += order_of_magnitude_;
      sec -= 1;
    }
    return std::make_tuple(sec, frac);
  }
};

/// Represents a year, month, day in a calendar.
struct Date {
  int year;
  unsigned month;
  unsigned day;
};

/// Represents a local time in a day, independent of any particular day.
struct Time {
  unsigned hour;
  unsigned minute;
  unsigned second;
};

/// Represents an ISO calendar date specified by year, week, and day of week.
struct ISOCalendar {
  int year;
  unsigned week;
  unsigned weekday;
};

/// Get the number of hours, minutes and seconds elapsed in the day
constexpr auto time_from_seconds(const int64_t seconds) noexcept -> Time {
  auto seconds_in_day = seconds % kSecondsInDay;
  if (seconds_in_day < 0) {
    seconds_in_day += kSecondsInDay;
  }
  const auto seconds_in_hour = seconds_in_day % kSecondsInHour;

  return {static_cast<unsigned>(seconds_in_day / kSecondsInHour),
          static_cast<unsigned>(seconds_in_hour / kSecondsInMinute),
          static_cast<unsigned>(seconds_in_hour % kSecondsInMinute)};
}

/// Get the date from the number of years since 1970-01-01
constexpr auto date_from_years(const int64_t years) -> Date {
  return {static_cast<int>(kEpoch + years), 1, 1};
}

/// Get the date from the number of months since 1970-01-01
constexpr auto date_from_months(const int64_t months) -> Date {
  auto year = months / kMonthsInYear;
  auto month = months % kMonthsInYear;
  if (month != 0 && months < 0) {
    --year;
    month += kMonthsInYear;
  }
  return {static_cast<int>(kEpoch + year), static_cast<unsigned>(month + 1), 1};
}

/// Gets year, month, day in civil calendar
constexpr auto date_from_days(int64_t days) noexcept -> Date {
  days += 719468LL;
  // era : 400 year period
  const auto era =
      static_cast<int>((days >= 0 ? days : days - 146096LL) / 146097LL);
  // day of era
  const auto doe = static_cast<unsigned>(days - era * 146097LL);  // [0, 146096]
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

/// Get the date from the number of weeks since 1970-01-01
constexpr auto date_from_weeks(const int64_t weeks) -> Date {
  return date_from_days(weeks * kDaysInWeek);
}

/// Get the date from the number of hours since 1970-01-01
constexpr auto datetime_from_hours(const int64_t hours)
    -> std::tuple<Date, Time> {
  auto days = hours / kHoursInDay;
  auto hour = hours % kHoursInDay;
  if (hour != 0 && hours < 0) {
    --days;
    hour += kHoursInDay;
  }
  return std::make_tuple(date_from_days(days),
                         Time{static_cast<unsigned>(hour), 0, 0});
}

/// Get the date from the number of minutes since 1970-01-01
constexpr auto datetime_from_minutes(const int64_t minutes)
    -> std::tuple<Date, Time> {
  auto days = minutes / kMinutesInDay;
  auto minute = minutes % kMinutesInDay;
  if (minute != 0 && minutes < 0) {
    --days;
    minute += kMinutesInDay;
  }
  return std::make_tuple(date_from_days(days),
                         time_from_seconds(minute * kSecondsInMinute));
}

/// Convert a datetime64 to a string
inline auto datetime64_to_string(const int64_t value, const DType &dtype)
    -> std::string {
  auto date = Date{};
  auto time = Time{};
  auto ss = std::stringstream{};

  switch (dtype.resolution()) {
    // Value is encoded as years elapsed since 1970.
    case DType::kYear:
      date = date_from_years(value);
      ss << date.year;
      break;

      // Value is encoded as months elapsed since 1970.
    case DType::kMonth:
      date = date_from_months(value);
      ss << date.year << "-" << std::setfill('0') << std::setw(2)
         << std::to_string(date.month);
      break;

      // Value is encoded as weeks elapsed since 1970.
    case DType::kWeek:
      date = date_from_weeks(value);
      ss << date.year << "-" << std::setfill('0') << std::setw(2) << date.month
         << "-" << std::setfill('0') << std::setw(2) << date.day;
      break;

      // Value is encoded as days elapsed since 1970.
    case DType::kDay:
      date = date_from_days(value);
      ss << date.year << "-" << std::setfill('0') << std::setw(2) << date.month
         << "-" << std::setfill('0') << std::setw(2) << date.day;
      break;

      // Value is encoded as hours elapsed since 1970.
    case DType::kHour:
      std::tie(date, time) = datetime_from_hours(value);
      ss << date.year << "-" << std::setfill('0') << std::setw(2) << date.month
         << "-" << std::setfill('0') << std::setw(2) << date.day << "T"
         << std::setfill('0') << std::setw(2) << time.hour;
      break;

      // Value is encoded as minutes elapsed since 1970.
    case DType::kMinute:
      std::tie(date, time) = datetime_from_minutes(value);
      ss << date.year << "-" << std::setfill('0') << std::setw(2) << date.month
         << "-" << std::setfill('0') << std::setw(2) << date.day << "T"
         << std::setfill('0') << std::setw(2) << time.hour << ":"
         << std::setfill('0') << std::setw(2) << time.minute;
      break;

    default:
      break;
  }

  // If resolution is one of the above, return the string.
  if (ss.tellp() != 0) {
    return ss.str();
  }

  // Value is encoded as fractional seconds elapsed since 1970 (the constructor
  // throws an exception if the resolution is invalid).
  auto frac = FractionalSeconds(dtype);
  auto [days, seconds, fractional] = frac.days_since_epoch(value);

  date = date_from_days(days);
  time = time_from_seconds(seconds);

  // Write the date and time.
  ss << date.year << "-" << std::setfill('0') << std::setw(2) << date.month
     << "-" << std::setfill('0') << std::setw(2) << date.day << "T"
     << std::setfill('0') << std::setw(2) << time.hour << ":"
     << std::setfill('0') << std::setw(2) << time.minute << ":"
     << std::setfill('0') << std::setw(2) << time.second;

  int ndigits = frac.ndigits();
  if (ndigits > 0) {
    ss << "." << std::setfill('0') << std::setw(ndigits)
       << std::to_string(fractional);
  }
  return ss.str();
}

/// True if leap year, else false.
constexpr auto is_leap_year(const int year) -> int {
  return (year & 3) == 0 && ((year % 25) != 0 || (year & 15) == 0);  // NOLINT
}

/// Get the number of days since the first January
constexpr auto days_since_january(const Date &date) -> unsigned {
  unsigned result = date.day - 1;

  if (date.month > 2) {
    result += is_leap_year(date.year);
  }

  for (size_t ix = 1; ix < date.month; ++ix) {
    result += kDaysInMonth[ix];
  }

  return result;
}

/// Get the week day of the week; Sunday is 0 ... Saturday is 6
constexpr auto weekday(const int64_t days) noexcept -> unsigned {
  return static_cast<unsigned>(days >= -4 ? (days + 4) % 7
                                          : (days + 5) % 7 + 6);
}

/// The number of days from the first day of the first ISO week of this year to
/// the year day *yday* with week day *wday*.  ISO weeks start on Monday; the
/// first ISO week has the year's first Thursday. *yday* may be as small as
/// *kYDayMinimum*.
constexpr auto iso_week_days(int yday, int wday) -> int {
  constexpr int big_enough_multiple_of_7 = (-kYDayMinimum / 7 + 2) * 7;
  return yday -
         (yday - wday + kIsoWeekFirstWDay + big_enough_multiple_of_7) % 7 +
         kIsoWeekFirstWDay - kIsoWeekStartWDay;
}

/// Return the ISO calendar
///
/// The first ISO week of the year is the (Mon-Sun) week
/// containing the year's first Thursday; everything else derives
/// from that.
///
///  The first week is 1; Monday is 1 ... Sunday is 7.
constexpr auto isocalendar(const int64_t days_since_epoch) -> ISOCalendar {
  auto date = date_from_days(days_since_epoch);
  auto yday = days_since_january(date);
  auto wday = weekday(days_since_epoch);
  auto days = iso_week_days(static_cast<int>(yday), static_cast<int>(wday));

  // This ISO week belongs to the previous year ?
  if (days < 0) {
    date.year--;
    days =
        iso_week_days(static_cast<int>(yday) + (365 + is_leap_year(date.year)),
                      static_cast<int>(wday));
  } else {
    int week_days =
        iso_week_days(static_cast<int>(yday) - (365 + is_leap_year(date.year)),
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

}  // namespace pyinterp::dateutils
