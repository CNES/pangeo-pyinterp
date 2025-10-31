// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include "pyinterp/detail/interpolation/interpolator.hpp"

template <typename T>
using Vector = Eigen::Array<T, Eigen::Dynamic, 1>;

TEST(Interpolator, Search) {
  Vector<double> x(10);
  x << 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;

  auto interpolator = pyinterp::detail::interpolation::Interpolator<double>();
  auto where = interpolator.search(x, 0.5);
  ASSERT_TRUE(where.has_value());
  auto [i0, i1] = *where;
  EXPECT_EQ(i0, 4);
  EXPECT_EQ(i1, 5);
  where = interpolator.search(x, 0.4);
  ASSERT_TRUE(where.has_value());
  std::tie(i0, i1) = *where;
  EXPECT_EQ(i0, 3);
  EXPECT_EQ(i1, 4);
  where = interpolator.search(x, 0.9);
  ASSERT_TRUE(where.has_value());
  std::tie(i0, i1) = *where;
  EXPECT_EQ(i0, 8);
  EXPECT_EQ(i1, 9);
  where = interpolator.search(x, 1.0);
  ASSERT_FALSE(where.has_value());
  where = interpolator.search(x, -1.0);
  ASSERT_FALSE(where.has_value());
}
