// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include "pyinterp/detail/math/linear.hpp"

namespace math = pyinterp::detail::math;

TEST(math_linear, linear) {
  /// https://en.wikipedia.org/wiki/Bilinear_interpolation
  EXPECT_DOUBLE_EQ(math::linear(14.5, 14.0, 15.0, 91.0, 210.0), 150.5);
  EXPECT_DOUBLE_EQ(math::linear(14.5, 14.0, 15.0, 162.0, 95.0), 128.5);

  auto y = math::linear<int64_t, double>(14, 13, 15, 91.0, 210.0);
  EXPECT_DOUBLE_EQ(y, 150.5);

  y = math::linear<int64_t, double>(14, 13, 15, 162.0, 95.0);
  EXPECT_DOUBLE_EQ(y, 128.5);
}
