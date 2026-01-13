// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/dateutils.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace pyinterp::dateutils {

// Checked arithmetic overflow detection
TEST(DateutilsCheckedArithmeticTest, AddOverflowPositive) {
  EXPECT_THROW(static_cast<void>(
                   detail::ckd_add(std::numeric_limits<int64_t>::max(), 1)),
               std::overflow_error);
}

TEST(DateutilsCheckedArithmeticTest, AddOverflowNegative) {
  EXPECT_THROW(static_cast<void>(
                   detail::ckd_add(std::numeric_limits<int64_t>::min(), -1)),
               std::overflow_error);
}

TEST(DateutilsCheckedArithmeticTest, MulOverflowPositive) {
  EXPECT_THROW(static_cast<void>(
                   detail::ckd_mul(std::numeric_limits<int64_t>::max(), 2)),
               std::overflow_error);
}

TEST(DateutilsCheckedArithmeticTest, MulOverflowCrossSign) {
  EXPECT_THROW(static_cast<void>(
                   detail::ckd_mul(std::numeric_limits<int64_t>::min(), 2)),
               std::overflow_error);
}

// Test fixture for DType tests
class DTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test valid datetime64 construction
TEST_F(DTypeTest, ConstructorDatetime64Valid) {
  EXPECT_NO_THROW(DType("datetime64[Y]"));
  EXPECT_NO_THROW(DType("datetime64[M]"));
  EXPECT_NO_THROW(DType("datetime64[W]"));
  EXPECT_NO_THROW(DType("datetime64[D]"));
  EXPECT_NO_THROW(DType("datetime64[h]"));
  EXPECT_NO_THROW(DType("datetime64[m]"));
  EXPECT_NO_THROW(DType("datetime64[s]"));
  EXPECT_NO_THROW(DType("datetime64[ms]"));
  EXPECT_NO_THROW(DType("datetime64[us]"));
  EXPECT_NO_THROW(DType("datetime64[ns]"));
  EXPECT_NO_THROW(DType("datetime64[ps]"));
  EXPECT_NO_THROW(DType("datetime64[fs]"));
  EXPECT_NO_THROW(DType("datetime64[as]"));
}

// Test valid timedelta64 construction
TEST_F(DTypeTest, ConstructorTimedelta64Valid) {
  EXPECT_NO_THROW(DType("timedelta64[Y]"));
  EXPECT_NO_THROW(DType("timedelta64[M]"));
  EXPECT_NO_THROW(DType("timedelta64[W]"));
  EXPECT_NO_THROW(DType("timedelta64[D]"));
  EXPECT_NO_THROW(DType("timedelta64[h]"));
  EXPECT_NO_THROW(DType("timedelta64[m]"));
  EXPECT_NO_THROW(DType("timedelta64[s]"));
  EXPECT_NO_THROW(DType("timedelta64[ms]"));
  EXPECT_NO_THROW(DType("timedelta64[us]"));
  EXPECT_NO_THROW(DType("timedelta64[ns]"));
  EXPECT_NO_THROW(DType("timedelta64[ps]"));
  EXPECT_NO_THROW(DType("timedelta64[fs]"));
  EXPECT_NO_THROW(DType("timedelta64[as]"));
}

// Test invalid dtype strings
TEST_F(DTypeTest, ConstructorInvalid) {
  EXPECT_THROW(DType("invalid"), std::invalid_argument);
  EXPECT_THROW(DType("datetime64"), std::invalid_argument);
  EXPECT_THROW(DType("datetime64[]"), std::invalid_argument);
  EXPECT_THROW(DType("datetime64[X]"), std::invalid_argument);
  EXPECT_THROW(DType("timedelta64[X]"), std::invalid_argument);
  EXPECT_THROW(DType("int64"), std::invalid_argument);
  EXPECT_THROW(DType(""), std::invalid_argument);
}

// Test constexpr constructor
TEST_F(DTypeTest, ConstexprConstructor) {
  constexpr DType dt(DType::DateType::kDatetime64, DType::Resolution::kSecond);
  EXPECT_EQ(dt.datetype(), DType::DateType::kDatetime64);
  EXPECT_EQ(dt.resolution(), DType::Resolution::kSecond);
}

// Test clock_name() for all resolutions
TEST_F(DTypeTest, ClockName) {
  EXPECT_EQ(DType("datetime64[Y]").clock_name(), "year");
  EXPECT_EQ(DType("datetime64[M]").clock_name(), "month");
  EXPECT_EQ(DType("datetime64[W]").clock_name(), "week");
  EXPECT_EQ(DType("datetime64[D]").clock_name(), "day");
  EXPECT_EQ(DType("datetime64[h]").clock_name(), "hour");
  EXPECT_EQ(DType("datetime64[m]").clock_name(), "minute");
  EXPECT_EQ(DType("datetime64[s]").clock_name(), "second");
  EXPECT_EQ(DType("datetime64[ms]").clock_name(), "millisecond");
  EXPECT_EQ(DType("datetime64[us]").clock_name(), "microsecond");
  EXPECT_EQ(DType("datetime64[ns]").clock_name(), "nanosecond");
  EXPECT_EQ(DType("datetime64[ps]").clock_name(), "picosecond");
  EXPECT_EQ(DType("datetime64[fs]").clock_name(), "femtosecond");
  EXPECT_EQ(DType("datetime64[as]").clock_name(), "attosecond");
}

// Test unit() for all resolutions
TEST_F(DTypeTest, Unit) {
  EXPECT_EQ(DType("datetime64[Y]").unit(), "Y");
  EXPECT_EQ(DType("datetime64[M]").unit(), "M");
  EXPECT_EQ(DType("datetime64[W]").unit(), "W");
  EXPECT_EQ(DType("datetime64[D]").unit(), "D");
  EXPECT_EQ(DType("datetime64[h]").unit(), "h");
  EXPECT_EQ(DType("datetime64[m]").unit(), "m");
  EXPECT_EQ(DType("datetime64[s]").unit(), "s");
  EXPECT_EQ(DType("datetime64[ms]").unit(), "ms");
  EXPECT_EQ(DType("datetime64[us]").unit(), "us");
  EXPECT_EQ(DType("datetime64[ns]").unit(), "ns");
  EXPECT_EQ(DType("datetime64[ps]").unit(), "ps");
  EXPECT_EQ(DType("datetime64[fs]").unit(), "fs");
  EXPECT_EQ(DType("datetime64[as]").unit(), "as");
}

// Test resolution() getter
TEST_F(DTypeTest, Resolution) {
  EXPECT_EQ(DType("datetime64[Y]").resolution(), DType::Resolution::kYear);
  EXPECT_EQ(DType("datetime64[M]").resolution(), DType::Resolution::kMonth);
  EXPECT_EQ(DType("datetime64[W]").resolution(), DType::Resolution::kWeek);
  EXPECT_EQ(DType("datetime64[D]").resolution(), DType::Resolution::kDay);
  EXPECT_EQ(DType("datetime64[h]").resolution(), DType::Resolution::kHour);
  EXPECT_EQ(DType("datetime64[m]").resolution(), DType::Resolution::kMinute);
  EXPECT_EQ(DType("datetime64[s]").resolution(), DType::Resolution::kSecond);
  EXPECT_EQ(DType("datetime64[ms]").resolution(),
            DType::Resolution::kMillisecond);
  EXPECT_EQ(DType("datetime64[us]").resolution(),
            DType::Resolution::kMicrosecond);
  EXPECT_EQ(DType("datetime64[ns]").resolution(),
            DType::Resolution::kNanosecond);
  EXPECT_EQ(DType("datetime64[ps]").resolution(),
            DType::Resolution::kPicosecond);
  EXPECT_EQ(DType("datetime64[fs]").resolution(),
            DType::Resolution::kFemtosecond);
  EXPECT_EQ(DType("datetime64[as]").resolution(),
            DType::Resolution::kAttosecond);
}

// Test datetype() getter
TEST_F(DTypeTest, DateType) {
  EXPECT_EQ(DType("datetime64[s]").datetype(), DType::DateType::kDatetime64);
  EXPECT_EQ(DType("timedelta64[s]").datetype(), DType::DateType::kTimedelta64);
}

// Test datetype_name()
TEST_F(DTypeTest, DateTypeName) {
  EXPECT_EQ(DType("datetime64[s]").datetype_name(), "datetime64");
  EXPECT_EQ(DType("timedelta64[s]").datetype_name(), "timedelta64");
}

// Test string conversion operator
TEST_F(DTypeTest, StringConversion) {
  EXPECT_EQ(std::string(DType("datetime64[Y]")), "datetime64[Y]");
  EXPECT_EQ(std::string(DType("datetime64[ms]")), "datetime64[ms]");
  EXPECT_EQ(std::string(DType("timedelta64[ns]")), "timedelta64[ns]");
  EXPECT_EQ(std::string(DType("timedelta64[D]")), "timedelta64[D]");
}

// Test order_of_magnitude() for valid resolutions
TEST_F(DTypeTest, OrderOfMagnitudeValid) {
  EXPECT_EQ(DType("datetime64[s]").order_of_magnitude(), kSecond);
  EXPECT_EQ(DType("datetime64[ms]").order_of_magnitude(), kMillisecond);
  EXPECT_EQ(DType("datetime64[us]").order_of_magnitude(), kMicrosecond);
  EXPECT_EQ(DType("datetime64[ns]").order_of_magnitude(), kNanosecond);
  EXPECT_EQ(DType("datetime64[ps]").order_of_magnitude(), kPicoSecond);
  EXPECT_EQ(DType("datetime64[fs]").order_of_magnitude(), kFemtoSecond);
  EXPECT_EQ(DType("datetime64[as]").order_of_magnitude(), kAttosecond);
}

// Test order_of_magnitude() throws for invalid resolutions
TEST_F(DTypeTest, OrderOfMagnitudeInvalid) {
  EXPECT_THROW((void)(DType("datetime64[Y]").order_of_magnitude()),
               std::invalid_argument);
  EXPECT_THROW((void)(DType("datetime64[M]").order_of_magnitude()),
               std::invalid_argument);
  EXPECT_THROW((void)(DType("datetime64[W]").order_of_magnitude()),
               std::invalid_argument);
  EXPECT_THROW((void)(DType("datetime64[D]").order_of_magnitude()),
               std::invalid_argument);
  EXPECT_THROW((void)(DType("datetime64[h]").order_of_magnitude()),
               std::invalid_argument);
  EXPECT_THROW((void)(DType("datetime64[m]").order_of_magnitude()),
               std::invalid_argument);
}

// Test comparison operators
TEST_F(DTypeTest, ComparisonOperators) {
  DType dt1("datetime64[s]");
  DType dt2("datetime64[s]");
  DType dt3("datetime64[ms]");
  DType dt4("timedelta64[s]");

  EXPECT_EQ(dt1, dt2);
  EXPECT_NE(dt1, dt3);
  EXPECT_NE(dt1, dt4);

  // Test spaceship operator
  EXPECT_TRUE(dt1 == dt2);
  EXPECT_FALSE(dt1 == dt3);
  EXPECT_FALSE(dt1 == dt4);
}

// Test as_timedelta64() conversion
TEST_F(DTypeTest, AsTimedelta64) {
  DType dt_datetime("datetime64[ns]");
  DType dt_converted = dt_datetime.as_timedelta64();

  EXPECT_EQ(dt_converted.datetype(), DType::DateType::kTimedelta64);
  EXPECT_EQ(dt_converted.resolution(), DType::Resolution::kNanosecond);
  EXPECT_EQ(dt_converted.unit(), "ns");

  DType dt_timedelta("timedelta64[ms]");
  DType dt_converted2 = dt_timedelta.as_timedelta64();

  EXPECT_EQ(dt_converted2.datetype(), DType::DateType::kTimedelta64);
  EXPECT_EQ(dt_converted2.resolution(), DType::Resolution::kMillisecond);
}

// Test that as_timedelta64() preserves resolution
TEST_F(DTypeTest, AsTimedelta64PreservesResolution) {
  std::vector<std::string> units = {"Y",  "M",  "W",  "D",  "h",  "m", "s",
                                    "ms", "us", "ns", "ps", "fs", "as"};

  for (const auto& unit : units) {
    DType original("datetime64[" + unit + "]");
    DType converted = original.as_timedelta64();

    EXPECT_EQ(converted.datetype(), DType::DateType::kTimedelta64);
    EXPECT_EQ(converted.resolution(), original.resolution());
    EXPECT_EQ(converted.unit(), unit);
  }
}

// Test the scale_factor() method
TEST_F(DTypeTest, Convert) {
  auto value = convert(55, DType("timedelta64[Y]"), DType("timedelta64[M]"));
  EXPECT_EQ(value, 55 * 12);
  value = convert(value, DType("timedelta64[M]"), DType("timedelta64[Y]"));

  EXPECT_EQ(value, 55);
  value = convert(value, DType("timedelta64[Y]"), DType("timedelta64[W]"));
  EXPECT_EQ(value, 2869);
  value = convert(value, DType("timedelta64[W]"), DType("timedelta64[Y]"));
  EXPECT_EQ(value, 54);  // Integer division

  value = convert(55, DType("timedelta64[Y]"), DType("timedelta64[D]"));
  EXPECT_EQ(value, 20089);
  value = convert(value, DType("timedelta64[D]"), DType("timedelta64[Y]"));
  EXPECT_EQ(value, 55);

  value = convert(value, DType("timedelta64[Y]"), DType("timedelta64[h]"));
  EXPECT_EQ(value, 482136);
  value = convert(value, DType("timedelta64[h]"), DType("timedelta64[Y]"));
  EXPECT_EQ(value, 55);

  value = convert(value, DType("timedelta64[Y]"), DType("timedelta64[s]"));
  EXPECT_EQ(value, 1'735'689'600);
  value = convert(value, DType("timedelta64[s]"), DType("timedelta64[Y]"));
  EXPECT_EQ(value, 55);

  value = convert(value, DType("timedelta64[Y]"), DType("timedelta64[ns]"));
  EXPECT_EQ(value, 1'735'689'600'000'000'000);
  value = convert(value, DType("timedelta64[ns]"), DType("timedelta64[Y]"));
  EXPECT_EQ(value, 55);
}

TEST_F(DTypeTest, ConvertOverflowToFinerResolution) {
  EXPECT_THROW(convert(std::numeric_limits<int64_t>::max(),
                       DType("datetime64[D]"), DType("datetime64[as]")),
               std::overflow_error);
}

// ============================================================================
// FractionalSeconds Tests
// ============================================================================

class FractionalSecondsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test constructor from DType
TEST_F(FractionalSecondsTest, ConstructorFromDType) {
  DType dt_s("datetime64[s]");
  EXPECT_NO_THROW({ FractionalSeconds fs(dt_s); });

  DType dt_ms("datetime64[ms]");
  EXPECT_NO_THROW({ FractionalSeconds fs(dt_ms); });

  DType dt_ns("datetime64[ns]");
  EXPECT_NO_THROW({ FractionalSeconds fs(dt_ns); });
}

// Test constructor from string
TEST_F(FractionalSecondsTest, ConstructorFromString) {
  EXPECT_NO_THROW(FractionalSeconds("datetime64[s]"));
  EXPECT_NO_THROW(FractionalSeconds("datetime64[ms]"));
  EXPECT_NO_THROW(FractionalSeconds("datetime64[us]"));
  EXPECT_NO_THROW(FractionalSeconds("datetime64[ns]"));
  EXPECT_NO_THROW(FractionalSeconds("datetime64[ps]"));
  EXPECT_NO_THROW(FractionalSeconds("datetime64[fs]"));
  EXPECT_NO_THROW(FractionalSeconds("datetime64[as]"));
}

// Test constructor throws for invalid resolutions
TEST_F(FractionalSecondsTest, ConstructorInvalidResolution) {
  EXPECT_THROW(FractionalSeconds("datetime64[Y]"), std::invalid_argument);
  EXPECT_THROW(FractionalSeconds("datetime64[M]"), std::invalid_argument);
  EXPECT_THROW(FractionalSeconds("datetime64[W]"), std::invalid_argument);
  EXPECT_THROW(FractionalSeconds("datetime64[D]"), std::invalid_argument);
  EXPECT_THROW(FractionalSeconds("datetime64[h]"), std::invalid_argument);
  EXPECT_THROW(FractionalSeconds("datetime64[m]"), std::invalid_argument);
}

// Test order_of_magnitude()
TEST_F(FractionalSecondsTest, OrderOfMagnitude) {
  EXPECT_EQ(FractionalSeconds("datetime64[s]").order_of_magnitude(), kSecond);
  EXPECT_EQ(FractionalSeconds("datetime64[ms]").order_of_magnitude(),
            kMillisecond);
  EXPECT_EQ(FractionalSeconds("datetime64[us]").order_of_magnitude(),
            kMicrosecond);
  EXPECT_EQ(FractionalSeconds("datetime64[ns]").order_of_magnitude(),
            kNanosecond);
  EXPECT_EQ(FractionalSeconds("datetime64[ps]").order_of_magnitude(),
            kPicoSecond);
  EXPECT_EQ(FractionalSeconds("datetime64[fs]").order_of_magnitude(),
            kFemtoSecond);
  EXPECT_EQ(FractionalSeconds("datetime64[as]").order_of_magnitude(),
            kAttosecond);
}

// Test resolution()
TEST_F(FractionalSecondsTest, Resolution) {
  EXPECT_EQ(FractionalSeconds("datetime64[s]").resolution(),
            DType::Resolution::kSecond);
  EXPECT_EQ(FractionalSeconds("datetime64[ms]").resolution(),
            DType::Resolution::kMillisecond);
  EXPECT_EQ(FractionalSeconds("datetime64[us]").resolution(),
            DType::Resolution::kMicrosecond);
  EXPECT_EQ(FractionalSeconds("datetime64[ns]").resolution(),
            DType::Resolution::kNanosecond);
  EXPECT_EQ(FractionalSeconds("datetime64[ps]").resolution(),
            DType::Resolution::kPicosecond);
  EXPECT_EQ(FractionalSeconds("datetime64[fs]").resolution(),
            DType::Resolution::kFemtosecond);
  EXPECT_EQ(FractionalSeconds("datetime64[as]").resolution(),
            DType::Resolution::kAttosecond);
}

// Test ndigits()
TEST_F(FractionalSecondsTest, NDigits) {
  EXPECT_EQ(FractionalSeconds("datetime64[s]").ndigits(), 0);
  EXPECT_EQ(FractionalSeconds("datetime64[ms]").ndigits(), 3);
  EXPECT_EQ(FractionalSeconds("datetime64[us]").ndigits(), 6);
  EXPECT_EQ(FractionalSeconds("datetime64[ns]").ndigits(), 9);
  EXPECT_EQ(FractionalSeconds("datetime64[ps]").ndigits(), 12);
  EXPECT_EQ(FractionalSeconds("datetime64[fs]").ndigits(), 15);
  EXPECT_EQ(FractionalSeconds("datetime64[as]").ndigits(), 18);
}

// Test days_since_epoch() for positive values
TEST_F(FractionalSecondsTest, DaysSinceEpochPositive) {
  FractionalSeconds frac_s("datetime64[s]");

  // 1 day in seconds (86400 seconds)
  auto [days, seconds, fractional] = frac_s.days_since_epoch(86400);
  EXPECT_EQ(days, 1);
  EXPECT_EQ(seconds, 0);  // 0 seconds within the day
  EXPECT_EQ(fractional, 0);

  // 1.5 days in seconds (129600 = 1 day + 43200 seconds)
  std::tie(days, seconds, fractional) = frac_s.days_since_epoch(129600);
  EXPECT_EQ(days, 1);
  EXPECT_EQ(seconds, 43200);  // 43200 seconds within the day (12 hours)
  EXPECT_EQ(fractional, 0);
}

// Test days_since_epoch() for negative values
TEST_F(FractionalSecondsTest, DaysSinceEpochNegative) {
  FractionalSeconds frac_s("datetime64[s]");

  // -1 day in seconds
  auto [days, seconds, fractional] = frac_s.days_since_epoch(-86400);
  EXPECT_EQ(days, -1);
  EXPECT_EQ(seconds, 0);  // 0 seconds within the day
  EXPECT_EQ(fractional, 0);
}

// Test days_since_epoch() with fractional seconds
TEST_F(FractionalSecondsTest, DaysSinceEpochWithFractional) {
  FractionalSeconds frac_ms("datetime64[ms]");

  // 1 day + 1 second + 500 milliseconds
  auto [days, seconds, fractional] = frac_ms.days_since_epoch(86401500);
  EXPECT_EQ(days, 1);
  EXPECT_EQ(seconds, 1);  // 1 second within the day
  EXPECT_EQ(fractional, 500);
}

// Test cast() method - upscaling
TEST_F(FractionalSecondsTest, CastUpscale) {
  FractionalSeconds frac_s("datetime64[s]");

  // Cast 1 second to milliseconds (1000ms)
  EXPECT_EQ(frac_s.cast(1, kMillisecond), 1000);

  // Cast 1 second to microseconds (1000000us)
  EXPECT_EQ(frac_s.cast(1, kMicrosecond), 1000000);
}

// Test cast() method - downscaling
TEST_F(FractionalSecondsTest, CastDownscale) {
  FractionalSeconds frac_ns("datetime64[ns]");

  // Cast 1000000000 nanoseconds to seconds (1s)
  EXPECT_EQ(frac_ns.cast(1000000000, kSecond), 1);

  // Cast 1000000 nanoseconds to milliseconds (1ms)
  EXPECT_EQ(frac_ns.cast(1000000, kMillisecond), 1);
}

TEST_F(FractionalSecondsTest, CastOverflowUpscale) {
  FractionalSeconds frac_s("datetime64[s]");

  // Upscaling a modest value to attoseconds should overflow int64
  EXPECT_THROW(static_cast<void>(frac_s.cast(10, kAttosecond)),
               std::overflow_error);
}

// ============================================================================
// Date/Time Conversion Function Tests
// ============================================================================

class DateTimeFunctions : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test time_from_seconds()
TEST_F(DateTimeFunctions, TimeFromSeconds) {
  auto time = time_from_seconds(0);
  EXPECT_EQ(time.hour, 0);
  EXPECT_EQ(time.minute, 0);
  EXPECT_EQ(time.second, 0);

  time = time_from_seconds(3661);  // 1h 1m 1s
  EXPECT_EQ(time.hour, 1);
  EXPECT_EQ(time.minute, 1);
  EXPECT_EQ(time.second, 1);

  time = time_from_seconds(86399);  // 23h 59m 59s
  EXPECT_EQ(time.hour, 23);
  EXPECT_EQ(time.minute, 59);
  EXPECT_EQ(time.second, 59);
}

// Test time_from_seconds() with negative values
TEST_F(DateTimeFunctions, TimeFromSecondsNegative) {
  auto time = time_from_seconds(-3600);  // -1 hour wraps to 23:00:00
  EXPECT_EQ(time.hour, 23);
  EXPECT_EQ(time.minute, 0);
  EXPECT_EQ(time.second, 0);
}

// Test date_from_years()
TEST_F(DateTimeFunctions, DateFromYears) {
  auto date = date_from_years(0);
  EXPECT_EQ(date.year, 1970);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 1);

  date = date_from_years(50);
  EXPECT_EQ(date.year, 2020);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 1);

  date = date_from_years(-1);
  EXPECT_EQ(date.year, 1969);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 1);
}

// Test date_from_months()
TEST_F(DateTimeFunctions, DateFromMonths) {
  auto date = date_from_months(0);
  EXPECT_EQ(date.year, 1970);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 1);

  date = date_from_months(12);
  EXPECT_EQ(date.year, 1971);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 1);

  date = date_from_months(13);
  EXPECT_EQ(date.year, 1971);
  EXPECT_EQ(date.month, 2);
  EXPECT_EQ(date.day, 1);

  date = date_from_months(-1);
  EXPECT_EQ(date.year, 1969);
  EXPECT_EQ(date.month, 12);
  EXPECT_EQ(date.day, 1);
}

// Test date_from_days()
TEST_F(DateTimeFunctions, DateFromDays) {
  auto date = date_from_days(0);
  EXPECT_EQ(date.year, 1970);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 1);

  date = date_from_days(365);
  EXPECT_EQ(date.year, 1971);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 1);

  date = date_from_days(31);
  EXPECT_EQ(date.year, 1970);
  EXPECT_EQ(date.month, 2);
  EXPECT_EQ(date.day, 1);

  date = date_from_days(-1);
  EXPECT_EQ(date.year, 1969);
  EXPECT_EQ(date.month, 12);
  EXPECT_EQ(date.day, 31);
}

// Test date_from_weeks()
TEST_F(DateTimeFunctions, DateFromWeeks) {
  auto date = date_from_weeks(0);
  EXPECT_EQ(date.year, 1970);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 1);

  date = date_from_weeks(1);
  EXPECT_EQ(date.year, 1970);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 8);

  date = date_from_weeks(52);
  EXPECT_EQ(date.year, 1970);
  EXPECT_EQ(date.month, 12);
  EXPECT_EQ(date.day, 31);
}

// Test datetime_from_hours()
TEST_F(DateTimeFunctions, DatetimeFromHours) {
  auto [date, time] = datetime_from_hours(0);
  EXPECT_EQ(date.year, 1970);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 1);
  EXPECT_EQ(time.hour, 0);
  EXPECT_EQ(time.minute, 0);
  EXPECT_EQ(time.second, 0);

  std::tie(date, time) = datetime_from_hours(25);
  EXPECT_EQ(date.year, 1970);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 2);
  EXPECT_EQ(time.hour, 1);
  EXPECT_EQ(time.minute, 0);
  EXPECT_EQ(time.second, 0);
}

// Test datetime_from_minutes()
TEST_F(DateTimeFunctions, DatetimeFromMinutes) {
  auto [date, time] = datetime_from_minutes(0);
  EXPECT_EQ(date.year, 1970);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 1);
  EXPECT_EQ(time.hour, 0);
  EXPECT_EQ(time.minute, 0);
  EXPECT_EQ(time.second, 0);

  std::tie(date, time) = datetime_from_minutes(1500);
  EXPECT_EQ(date.year, 1970);
  EXPECT_EQ(date.month, 1);
  EXPECT_EQ(date.day, 2);
  EXPECT_EQ(time.hour, 1);
  EXPECT_EQ(time.minute, 0);
  EXPECT_EQ(time.second, 0);
}

// Test is_leap_year()
TEST_F(DateTimeFunctions, IsLeapYear) {
  EXPECT_FALSE(is_leap_year(1970));
  EXPECT_TRUE(is_leap_year(1972));
  EXPECT_FALSE(is_leap_year(1900));
  EXPECT_TRUE(is_leap_year(2000));
  EXPECT_TRUE(is_leap_year(2004));
  EXPECT_FALSE(is_leap_year(2001));
}

// Test days_since_january()
TEST_F(DateTimeFunctions, DaysSinceJanuary) {
  EXPECT_EQ(days_since_january({1970, 1, 1}), 0);
  EXPECT_EQ(days_since_january({1970, 1, 31}), 30);
  EXPECT_EQ(days_since_january({1970, 2, 1}), 31);
  EXPECT_EQ(days_since_january({1970, 12, 31}), 364);

  // Leap year
  EXPECT_EQ(days_since_january({1972, 3, 1}), 60);
}

// Test weekday()
TEST_F(DateTimeFunctions, Weekday) {
  EXPECT_EQ(weekday(0), 4);   // 1970-01-01 was Thursday
  EXPECT_EQ(weekday(1), 5);   // 1970-01-02 was Friday
  EXPECT_EQ(weekday(2), 6);   // 1970-01-03 was Saturday
  EXPECT_EQ(weekday(3), 0);   // 1970-01-04 was Sunday
  EXPECT_EQ(weekday(-1), 3);  // 1969-12-31 was Wednesday
}

// Test isocalendar()
TEST_F(DateTimeFunctions, ISOCalendar) {
  // 1970-01-01 was Thursday, which is in ISO week 1 of 1970
  auto iso = isocalendar(0);
  EXPECT_EQ(iso.year, 1970);
  EXPECT_EQ(iso.week, 1);
  EXPECT_EQ(iso.weekday, 4);  // Thursday

  // Test a known ISO week
  iso = isocalendar(7);  // 1970-01-08
  EXPECT_EQ(iso.year, 1970);
  EXPECT_EQ(iso.week, 2);
  EXPECT_EQ(iso.weekday, 4);  // Thursday
}

// Test datetime64_to_string() for years
TEST_F(DateTimeFunctions, Datetime64ToStringYears) {
  DType dtype("datetime64[Y]");
  EXPECT_EQ(datetime64_to_string(0, dtype), "1970");
  EXPECT_EQ(datetime64_to_string(50, dtype), "2020");
  EXPECT_EQ(datetime64_to_string(-1, dtype), "1969");
}

// Test datetime64_to_string() for months
TEST_F(DateTimeFunctions, Datetime64ToStringMonths) {
  DType dtype("datetime64[M]");
  EXPECT_EQ(datetime64_to_string(0, dtype), "1970-01");
  EXPECT_EQ(datetime64_to_string(12, dtype), "1971-01");
  EXPECT_EQ(datetime64_to_string(13, dtype), "1971-02");
}

// Test datetime64_to_string() for days
TEST_F(DateTimeFunctions, Datetime64ToStringDays) {
  DType dtype("datetime64[D]");
  EXPECT_EQ(datetime64_to_string(0, dtype), "1970-01-01");
  EXPECT_EQ(datetime64_to_string(365, dtype), "1971-01-01");
  EXPECT_EQ(datetime64_to_string(31, dtype), "1970-02-01");
}

// Test datetime64_to_string() for weeks
TEST_F(DateTimeFunctions, Datetime64ToStringWeeks) {
  DType dtype("datetime64[W]");
  EXPECT_EQ(datetime64_to_string(0, dtype), "1970-01-01");
  EXPECT_EQ(datetime64_to_string(1, dtype), "1970-01-08");
}

// Test datetime64_to_string() for hours
TEST_F(DateTimeFunctions, Datetime64ToStringHours) {
  DType dtype("datetime64[h]");
  EXPECT_EQ(datetime64_to_string(0, dtype), "1970-01-01T00");
  EXPECT_EQ(datetime64_to_string(25, dtype), "1970-01-02T01");
}

// Test datetime64_to_string() for minutes
TEST_F(DateTimeFunctions, Datetime64ToStringMinutes) {
  DType dtype("datetime64[m]");
  EXPECT_EQ(datetime64_to_string(0, dtype), "1970-01-01T00:00");
  EXPECT_EQ(datetime64_to_string(61, dtype), "1970-01-01T01:01");
}

// Test datetime64_to_string() for seconds
TEST_F(DateTimeFunctions, Datetime64ToStringSeconds) {
  DType dtype("datetime64[s]");
  EXPECT_EQ(datetime64_to_string(0, dtype), "1970-01-01T00:00:00");
  EXPECT_EQ(datetime64_to_string(3661, dtype), "1970-01-01T01:01:01");
}

// Test datetime64_to_string() for milliseconds
TEST_F(DateTimeFunctions, Datetime64ToStringMilliseconds) {
  DType dtype("datetime64[ms]");
  EXPECT_EQ(datetime64_to_string(0, dtype), "1970-01-01T00:00:00.000");
  EXPECT_EQ(datetime64_to_string(1500, dtype), "1970-01-01T00:00:01.500");
}

// Test datetime64_to_string() for microseconds
TEST_F(DateTimeFunctions, Datetime64ToStringMicroseconds) {
  DType dtype("datetime64[us]");
  EXPECT_EQ(datetime64_to_string(0, dtype), "1970-01-01T00:00:00.000000");
  EXPECT_EQ(datetime64_to_string(1500000, dtype), "1970-01-01T00:00:01.500000");
}

// Test datetime64_to_string() for nanoseconds
TEST_F(DateTimeFunctions, Datetime64ToStringNanoseconds) {
  DType dtype("datetime64[ns]");
  EXPECT_EQ(datetime64_to_string(0, dtype), "1970-01-01T00:00:00.000000000");
  EXPECT_EQ(datetime64_to_string(1500000000, dtype),
            "1970-01-01T00:00:01.500000000");
}

// Test timedelta64_to_string() for years
TEST_F(DateTimeFunctions, Timedelta64ToStringYears) {
  DType dtype("timedelta64[Y]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 years");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 years");
  EXPECT_EQ(timedelta64_to_string(50, dtype), "50 years");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 years");
  EXPECT_EQ(timedelta64_to_string(-50, dtype), "-50 years");
}

// Test timedelta64_to_string() for months
TEST_F(DateTimeFunctions, Timedelta64ToStringMonths) {
  DType dtype("timedelta64[M]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 months");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 months");
  EXPECT_EQ(timedelta64_to_string(12, dtype), "12 months");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 months");
  EXPECT_EQ(timedelta64_to_string(-12, dtype), "-12 months");
}

// Test timedelta64_to_string() for weeks
TEST_F(DateTimeFunctions, Timedelta64ToStringWeeks) {
  DType dtype("timedelta64[W]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 weeks");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 weeks");
  EXPECT_EQ(timedelta64_to_string(52, dtype), "52 weeks");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 weeks");
  EXPECT_EQ(timedelta64_to_string(-52, dtype), "-52 weeks");
}

// Test timedelta64_to_string() for days
TEST_F(DateTimeFunctions, Timedelta64ToStringDays) {
  DType dtype("timedelta64[D]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 days");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 days");
  EXPECT_EQ(timedelta64_to_string(365, dtype), "365 days");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 days");
  EXPECT_EQ(timedelta64_to_string(-365, dtype), "-365 days");
}

// Test timedelta64_to_string() for hours
TEST_F(DateTimeFunctions, Timedelta64ToStringHours) {
  DType dtype("timedelta64[h]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 hours");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 hours");
  EXPECT_EQ(timedelta64_to_string(24, dtype), "24 hours");
  EXPECT_EQ(timedelta64_to_string(25, dtype), "25 hours");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 hours");
  EXPECT_EQ(timedelta64_to_string(-25, dtype), "-25 hours");
}

// Test timedelta64_to_string() for minutes
TEST_F(DateTimeFunctions, Timedelta64ToStringMinutes) {
  DType dtype("timedelta64[m]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 minutes");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 minutes");
  EXPECT_EQ(timedelta64_to_string(60, dtype), "60 minutes");
  EXPECT_EQ(timedelta64_to_string(1440, dtype), "1440 minutes");
  EXPECT_EQ(timedelta64_to_string(1500, dtype), "1500 minutes");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 minutes");
  EXPECT_EQ(timedelta64_to_string(-61, dtype), "-61 minutes");
}

// Test timedelta64_to_string() for seconds
TEST_F(DateTimeFunctions, Timedelta64ToStringSeconds) {
  DType dtype("timedelta64[s]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 seconds");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 seconds");
  EXPECT_EQ(timedelta64_to_string(3661, dtype), "3661 seconds");
  EXPECT_EQ(timedelta64_to_string(86400, dtype), "86400 seconds");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 seconds");
  EXPECT_EQ(timedelta64_to_string(-3661, dtype), "-3661 seconds");
}

// Test timedelta64_to_string() for milliseconds
TEST_F(DateTimeFunctions, Timedelta64ToStringMilliseconds) {
  DType dtype("timedelta64[ms]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 milliseconds");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 milliseconds");
  EXPECT_EQ(timedelta64_to_string(1500, dtype), "1500 milliseconds");
  EXPECT_EQ(timedelta64_to_string(86400000, dtype), "86400000 milliseconds");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 milliseconds");
  EXPECT_EQ(timedelta64_to_string(-1500, dtype), "-1500 milliseconds");
}

// Test timedelta64_to_string() for microseconds
TEST_F(DateTimeFunctions, Timedelta64ToStringMicroseconds) {
  DType dtype("timedelta64[us]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 microseconds");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 microseconds");
  EXPECT_EQ(timedelta64_to_string(1500000, dtype), "1500000 microseconds");
  EXPECT_EQ(timedelta64_to_string(86400000000, dtype),
            "86400000000 microseconds");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 microseconds");
  EXPECT_EQ(timedelta64_to_string(-1500000, dtype), "-1500000 microseconds");
}

// Test timedelta64_to_string() for nanoseconds
TEST_F(DateTimeFunctions, Timedelta64ToStringNanoseconds) {
  DType dtype("timedelta64[ns]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 nanoseconds");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 nanoseconds");
  EXPECT_EQ(timedelta64_to_string(1500000000, dtype), "1500000000 nanoseconds");
  EXPECT_EQ(timedelta64_to_string(86400000000000, dtype),
            "86400000000000 nanoseconds");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 nanoseconds");
  EXPECT_EQ(timedelta64_to_string(-1500000000, dtype),
            "-1500000000 nanoseconds");
}

// Test timedelta64_to_string() for picoseconds
TEST_F(DateTimeFunctions, Timedelta64ToStringPicoseconds) {
  DType dtype("timedelta64[ps]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 picoseconds");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 picoseconds");
  EXPECT_EQ(timedelta64_to_string(1500000000000, dtype),
            "1500000000000 picoseconds");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 picoseconds");
}

// Test timedelta64_to_string() for femtoseconds
TEST_F(DateTimeFunctions, Timedelta64ToStringFemtoseconds) {
  DType dtype("timedelta64[fs]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 femtoseconds");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 femtoseconds");
  EXPECT_EQ(timedelta64_to_string(1500000000000000, dtype),
            "1500000000000000 femtoseconds");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 femtoseconds");
}

// Test timedelta64_to_string() for attoseconds
TEST_F(DateTimeFunctions, Timedelta64ToStringAttoseconds) {
  DType dtype("timedelta64[as]");
  EXPECT_EQ(timedelta64_to_string(0, dtype), "0 attoseconds");
  EXPECT_EQ(timedelta64_to_string(1, dtype), "1 attoseconds");
  EXPECT_EQ(timedelta64_to_string(1500000000000000000, dtype),
            "1500000000000000000 attoseconds");
  EXPECT_EQ(timedelta64_to_string(-1, dtype), "-1 attoseconds");
}

// Test timedelta64_to_string() with large values
TEST_F(DateTimeFunctions, Timedelta64ToStringLargeValues) {
  DType dtype_days("timedelta64[D]");
  EXPECT_EQ(timedelta64_to_string(1000000, dtype_days), "1000000 days");

  DType dtype_seconds("timedelta64[s]");
  EXPECT_EQ(timedelta64_to_string(86400000, dtype_seconds), "86400000 seconds");
}

// Test timedelta64_to_string() with zero fractional parts
TEST_F(DateTimeFunctions, Timedelta64ToStringZeroFractional) {
  DType dtype_ms("timedelta64[ms]");
  EXPECT_EQ(timedelta64_to_string(1000, dtype_ms), "1000 milliseconds");

  DType dtype_ns("timedelta64[ns]");
  EXPECT_EQ(timedelta64_to_string(1000000000, dtype_ns),
            "1000000000 nanoseconds");
}

}  // namespace pyinterp::dateutils
