// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/math/interpolate/linear.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

namespace pyinterp::math::interpolate {

// =============================================================================
// Typed test fixture to avoid duplication across float/double
// =============================================================================

template <typename U>
class LinearTest : public ::testing::Test {
 protected:
  static constexpr U kX0 = static_cast<U>(0);
  static constexpr U kX1 = static_cast<U>(10);
  static constexpr U kY0 = static_cast<U>(100);
  static constexpr U kY1 = static_cast<U>(200);

  static constexpr auto kTol() -> U {
    return static_cast<U>(10) * std::numeric_limits<U>::epsilon();
  }
};

using FloatingTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LinearTest, FloatingTypes);

// Midpoint
TYPED_TEST(LinearTest, Midpoint) {
  auto x = static_cast<TypeParam>(5);
  auto y = linear<TypeParam, TypeParam>(x, this->kX0, this->kX1, this->kY0,
                                        this->kY1);
  EXPECT_NEAR(y, static_cast<TypeParam>(150), this->kTol());
}

// Quarter point
TYPED_TEST(LinearTest, QuarterPoint) {
  auto x = static_cast<TypeParam>(2.5);
  auto y = linear<TypeParam, TypeParam>(x, this->kX0, this->kX1, this->kY0,
                                        this->kY1);
  EXPECT_NEAR(y, static_cast<TypeParam>(125), this->kTol());
}

// Exact match with first endpoint
TYPED_TEST(LinearTest, ExactFirst) {
  auto y = linear<TypeParam, TypeParam>(this->kX0, this->kX0, this->kX1,
                                        this->kY0, this->kY1);
  EXPECT_NEAR(y, this->kY0, this->kTol());
}

// Exact match with second endpoint
TYPED_TEST(LinearTest, ExactSecond) {
  auto y = linear<TypeParam, TypeParam>(this->kX1, this->kX0, this->kX1,
                                        this->kY0, this->kY1);
  EXPECT_NEAR(y, this->kY1, this->kTol());
}

// Division by zero with x equal to x0
TYPED_TEST(LinearTest, DivisionByZeroExactMatch) {
  const auto x0 = static_cast<TypeParam>(5);
  const auto x1 = static_cast<TypeParam>(5);
  const auto y0 = static_cast<TypeParam>(100);
  const auto y1 = static_cast<TypeParam>(200);
  auto y = linear<TypeParam, TypeParam>(x0, x0, x1, y0, y1);
  EXPECT_NEAR(y, y0, this->kTol());
}

// Division by zero with x not equal to x0
TYPED_TEST(LinearTest, DivisionByZeroNoMatch) {
  const auto x0 = static_cast<TypeParam>(5);
  const auto x1 = static_cast<TypeParam>(5);
  const auto y0 = static_cast<TypeParam>(100);
  const auto y1 = static_cast<TypeParam>(200);
  auto y =
      linear<TypeParam, TypeParam>(static_cast<TypeParam>(7), x0, x1, y0, y1);
  EXPECT_TRUE(std::isnan(y));
}

// =============================================================================
// Mixed coordinate/value types (T != U)
// =============================================================================

TEST(LinearTestMixedTypes, MixedTypes) {
  constexpr int32_t kX0 = 0;
  constexpr int32_t kX1 = 10;
  constexpr float kY0 = 100.0f;
  constexpr float kY1 = 200.0f;

  // Midpoint
  {
    int32_t x = 5;
    auto y = linear<int32_t, float>(x, kX0, kX1, kY0, kY1);
    EXPECT_FLOAT_EQ(y, 150.0f);
  }

  // Quarter point (note: integer x truncates to 2, so 20% of segment)
  {
    int32_t x = 2;
    auto y = linear<int32_t, float>(x, kX0, kX1, kY0, kY1);
    EXPECT_FLOAT_EQ(y, 120.0f);
  }

  // Exact matches
  {
    auto y0 = linear<int32_t, float>(kX0, kX0, kX1, kY0, kY1);
    auto y1 = linear<int32_t, float>(kX1, kX0, kX1, kY0, kY1);
    EXPECT_FLOAT_EQ(y0, kY0);
    EXPECT_FLOAT_EQ(y1, kY1);
  }
}

}  // namespace pyinterp::math::interpolate
