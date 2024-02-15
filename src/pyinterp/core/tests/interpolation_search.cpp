// Copyright (c) 2024 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include "pyinterp/detail/interpolation/interpolator.hpp"

template <typename T>
using Vector = Eigen::Array<T, Eigen::Dynamic, 1>;

TEST(Interpolator, search) {
  Vector<double> x(10);
  x << 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;

  auto interpolator = pyinterp::detail::interpolation::Interpolator<double>();
  auto where = interpolator.search(x, 0.5);
  ASSERT_TRUE(where.has_value());
  auto [i0, i1] = *where;
  EXPECT_EQ(i0, 4);
  EXPECT_EQ(i1, 5);
  auto index = static_cast<Eigen::Index>(5);
  where = interpolator.search(x, 0.5, &index);
  ASSERT_TRUE(where.has_value());
  auto [i02, i12] = *where;
  EXPECT_EQ(i02, 4);
  EXPECT_EQ(i12, 5);
  index = 6;
  where = interpolator.search(x, 0.5, &index);
  ASSERT_TRUE(where.has_value());
  index = 7;
  where = interpolator.search(x, 0.5, &index);
  ASSERT_TRUE(where.has_value());
  where = interpolator.search(x, 0.0);
  ASSERT_TRUE(where.has_value());
  EXPECT_EQ(where->first, 0);
  EXPECT_EQ(where->second, 1);
  where = interpolator.search(x, 0.9);
  ASSERT_TRUE(where.has_value());
  EXPECT_EQ(where->first, 8);
  EXPECT_EQ(where->second, 9);
  index = 8;
  where = interpolator.search(x, 0.9, &index);
  ASSERT_TRUE(where.has_value());
  EXPECT_EQ(where->first, 8);
  EXPECT_EQ(where->second, 9);
  index = 9;
  where = interpolator.search(x, 0.9, &index);
  ASSERT_TRUE(where.has_value());
  EXPECT_EQ(where->first, 8);
  EXPECT_EQ(where->second, 9);
  index = 10;
  where = interpolator.search(x, 0.9, &index);
  ASSERT_TRUE(where.has_value());
  index = 11;
  where = interpolator.search(x, 0.9, &index);
  ASSERT_FALSE(where.has_value());
  where = interpolator.search(x, -0.1);
  ASSERT_FALSE(where.has_value());
  where = interpolator.search(x, 1.0);
}
