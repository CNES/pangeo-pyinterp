#include "pyinterp/detail/math/linear.hpp"
#include <boost/geometry.hpp>
#include <gtest/gtest.h>

namespace math = pyinterp::detail::math;

TEST(math_linear, linear) {
  /// https://en.wikipedia.org/wiki/Bilinear_interpolation
  EXPECT_DOUBLE_EQ(math::linear(14.5, 14.0, 15.0, 91.0, 210.0), 150.5);
  EXPECT_DOUBLE_EQ(math::linear(14.5, 14.0, 15.0, 162.0, 95.0), 128.5);
}

