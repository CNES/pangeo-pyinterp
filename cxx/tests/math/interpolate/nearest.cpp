// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/math/interpolate/nearest.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace pyinterp::math::interpolate {

// =============================================================================
// Typed test fixture
// =============================================================================

template <typename T>
class NearestTest : public ::testing::Test {
 protected:
  static constexpr T kX0 = static_cast<T>(0);
  static constexpr T kX1 = static_cast<T>(10);
  static constexpr T kY0 = static_cast<T>(100);
  static constexpr T kY1 = static_cast<T>(200);
};

using NumericTypes = ::testing::Types<int32_t, int64_t, float, double>;
TYPED_TEST_SUITE(NearestTest, NumericTypes);

// Closer to first endpoint returns y0
TYPED_TEST(NearestTest, CloserToFirst) {
  auto x = static_cast<TypeParam>(2.9);  // distance to 0 is 2.9, to 10 is 7.1
  auto y = nearest<TypeParam>(x, this->kX0, this->kX1, this->kY0, this->kY1);
  EXPECT_EQ(y, this->kY0);
}

// Closer to second endpoint returns y1
TYPED_TEST(NearestTest, CloserToSecond) {
  auto x = static_cast<TypeParam>(8.2);  // distance to 0 is 8.2, to 10 is 1.8
  auto y = nearest<TypeParam>(x, this->kX0, this->kX1, this->kY0, this->kY1);
  EXPECT_EQ(y, this->kY1);
}

// Exact match with first endpoint returns y0
TYPED_TEST(NearestTest, ExactFirst) {
  auto y =
      nearest<TypeParam>(this->kX0, this->kX0, this->kX1, this->kY0, this->kY1);
  EXPECT_EQ(y, this->kY0);
}

// Exact match with second endpoint returns y1
TYPED_TEST(NearestTest, ExactSecond) {
  auto y =
      nearest<TypeParam>(this->kX1, this->kX0, this->kX1, this->kY0, this->kY1);
  EXPECT_EQ(y, this->kY1);
}

// Tie: equidistant should pick y1 (implementation uses < not <=)
TYPED_TEST(NearestTest, TieChoosesSecond) {
  auto x = static_cast<TypeParam>(5);  // equidistant from 0 and 10
  auto y = nearest<TypeParam>(x, this->kX0, this->kX1, this->kY0, this->kY1);
  EXPECT_EQ(y, this->kY1);
}

// Negative coordinates still choose nearest correctly
TYPED_TEST(NearestTest, NegativeCoordinates) {
  auto x = static_cast<TypeParam>(-3);  // closer to 0 than to 10
  auto y = nearest<TypeParam>(x, this->kX0, this->kX1, this->kY0, this->kY1);
  EXPECT_EQ(y, this->kY0);
}

// Outside upper bound chooses closest end
TYPED_TEST(NearestTest, AboveUpperBound) {
  auto x = static_cast<TypeParam>(17);  // closer to 10
  auto y = nearest<TypeParam>(x, this->kX0, this->kX1, this->kY0, this->kY1);
  EXPECT_EQ(y, this->kY1);
}

}  // namespace pyinterp::math::interpolate
