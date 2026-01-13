// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/math/temporal_axis.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp {

TEST(TemporalAxis, RegularWithoutPeriod) {
  auto dtype = dateutils::DType(dateutils::DType::DateType::kDatetime64,
                                dateutils::DType::Resolution::kSecond);
  auto axis = math::TemporalAxis(dtype,
                                 946684800,  // 2000-01-01 00:00:00
                                 978220800,  // 2001-12-31 00:00:00
                                 366, 0, std::nullopt);

  EXPECT_EQ(axis.front(), 946684800);
  EXPECT_EQ(axis.back(), 978220800);
  EXPECT_EQ(axis.min_value(), 946684800);
  EXPECT_EQ(axis.max_value(), 978220800);
  EXPECT_EQ(axis.increment(), 86400);  // One day in seconds
  EXPECT_EQ(axis.size(), 366);
  EXPECT_FALSE(axis.is_periodic());
  EXPECT_EQ(axis.dtype(), dtype);
  auto repr = std::string(axis);
  EXPECT_EQ(repr,
            R"(TemporalAxis[datetime64[s]](regular)
  range: [2000-01-01T00:00:00, 2000-12-31T00:00:00]
  step: 86400 seconds
  size: 366)");

  auto state_reader = serialization::Reader(axis.pack());
  auto other = math::TemporalAxis::unpack(state_reader);
  EXPECT_EQ(axis, other);
}

TEST(TemporalAxis, RegularWithPeriod) {
  auto dtype = dateutils::DType(dateutils::DType::DateType::kTimedelta64,
                                dateutils::DType::Resolution::kMinute);
  auto axis = math::TemporalAxis(dtype,
                                 0,     // 0 minutes
                                 1439,  // 23 hours and 59 minutes
                                 1440,  // One day in minutes
                                 0,
                                 1440);  // Period of one day in minutes;

  EXPECT_EQ(axis.front(), 0);
  EXPECT_EQ(axis.back(), 1439);
  EXPECT_EQ(axis.min_value(), 0);
  EXPECT_EQ(axis.max_value(), 1439);
  EXPECT_EQ(axis.increment(), 1);  // One minute
  EXPECT_EQ(axis.size(), 1440);
  EXPECT_TRUE(axis.is_periodic());
  EXPECT_EQ(axis.dtype(), dtype);
  auto repr = std::string(axis);
  EXPECT_EQ(repr,
            R"(TemporalAxis[timedelta64[m]](regular, period=1440 minutes)
  range: [0, 1439]
  step: 1 minutes
  size: 1440)");

  auto state_reader = serialization::Reader(axis.pack());
  auto other = math::TemporalAxis::unpack(state_reader);
  EXPECT_EQ(axis, other);
}

TEST(TemporalAxis, IrregularAxis) {
  Eigen::Vector<int64_t, 10> values{10957, 11048, 11139, 11230, 11322,
                                    11413, 11504, 11595, 11688, 11781};
  // Corresponds to 2000-01-01, 2000-04-01, 2000-07-01, 2000-10-01, 2001-01-01,
  // 2001-04-01, 2001-07-01, 2001-10-01, 2002-01-01, 2002-04-04
  auto dtype = dateutils::DType(dateutils::DType::DateType::kDatetime64,
                                dateutils::DType::Resolution::kDay);
  auto axis = math::TemporalAxis(dtype, values, 1, std::nullopt);
  EXPECT_EQ(axis.front(), 10957);
  EXPECT_EQ(axis.back(), 11781);
  EXPECT_EQ(axis.min_value(), 10957);
  EXPECT_EQ(axis.max_value(), 11781);
  EXPECT_EQ(axis.size(), values.size());
  EXPECT_FALSE(axis.is_periodic());
  auto repr = std::string(axis);
  EXPECT_EQ(repr,
            R"(TemporalAxis[datetime64[D]](irregular)
  values: [2000-01-01, 2000-04-01, 2000-07-01, ..., 2001-09-30, 2002-01-01, 2002-04-04]
  size: 10)");

  auto state_reader = serialization::Reader(axis.pack());
  auto other = math::TemporalAxis::unpack(state_reader);
  EXPECT_EQ(axis, other);
}

}  // namespace pyinterp
