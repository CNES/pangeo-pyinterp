// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>
#include <boost/geometry.hpp>
#include "pyinterp/detail/math/linear.hpp"

namespace math = pyinterp::detail::math;

TEST(math_linear, linear) {
  /// https://en.wikipedia.org/wiki/Bilinear_interpolation
  EXPECT_DOUBLE_EQ(math::linear(14.5, 14.0, 15.0, 91.0, 210.0), 150.5);
  EXPECT_DOUBLE_EQ(math::linear(14.5, 14.0, 15.0, 162.0, 95.0), 128.5);
}
