// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>
#include "pyinterp/detail/axis/container.hpp"

namespace container = pyinterp::detail::axis::container;

TEST(axis_container, undefined) {
  // undefined axis
  auto a1 = container::Undefined();
  a1.flip();
  EXPECT_TRUE(std::isnan(a1.front()));
  EXPECT_TRUE(std::isnan(a1.back()));
  EXPECT_TRUE(std::isnan(a1.min_value()));
  EXPECT_TRUE(std::isnan(a1.max_value()));
  EXPECT_EQ(a1.size(), 0);
  EXPECT_EQ(a1.find_index(360, true), -1);
  EXPECT_EQ(a1.find_index(360, false), -1);
  EXPECT_EQ(a1, a1);
}

TEST(axis_container, irregular) {
  // irregular axis
  auto values = std::vector<double>{0, 1, 4, 8, 20};
  auto a1 = container::Irregular(
      Eigen::Map<Eigen::VectorXd>(values.data(), values.size()));
  EXPECT_EQ(a1.front(), 0);
  EXPECT_EQ(a1.back(), 20);
  EXPECT_EQ(a1.min_value(), 0);
  EXPECT_EQ(a1.max_value(), 20);
  EXPECT_EQ(a1.coordinate_value(2), 4);
  EXPECT_EQ(a1.find_index(8, false), 3);
  EXPECT_EQ(a1.find_index(8.3, false), 3);
  EXPECT_EQ(a1.find_index(30, true), 4);
  EXPECT_EQ(a1.find_index(20.1, true), 4);
  EXPECT_EQ(a1.find_index(30, false), -1);
  EXPECT_EQ(a1.size(), 5);
  EXPECT_EQ(a1, a1);
  a1.flip();
  EXPECT_EQ(a1.front(), 20);
  EXPECT_EQ(a1.back(), 0);
  EXPECT_EQ(a1.min_value(), 0);
  EXPECT_EQ(a1.max_value(), 20);
  EXPECT_EQ(a1.coordinate_value(2), 4);
  EXPECT_EQ(a1.find_index(8, false), 1);
  EXPECT_EQ(a1.find_index(8.3, false), 1);
  EXPECT_EQ(a1.find_index(30, true), 0);
  EXPECT_EQ(a1.find_index(20.1, true), 0);
  EXPECT_EQ(a1.find_index(30, false), -1);
  EXPECT_EQ(a1.size(), 5);
  EXPECT_EQ(a1, a1);
  values = std::vector<double>{0, 1};
  auto a2 = container::Irregular(
      Eigen::Map<Eigen::VectorXd>(values.data(), values.size()));
  EXPECT_FALSE(a1 == a2);
  EXPECT_FALSE(a1 == container::Undefined());
}

TEST(axis_container, regular) {
  // regular axis
  EXPECT_THROW(container::Regular(0, 359, 0), std::invalid_argument);
  auto a1 = container::Regular(0, 359, 360);
  EXPECT_EQ(a1.front(), 0);
  EXPECT_EQ(a1.back(), 359);
  EXPECT_EQ(a1.min_value(), 0);
  EXPECT_EQ(a1.max_value(), 359);
  EXPECT_EQ(a1.coordinate_value(2), 2);
  EXPECT_EQ(a1.find_index(180, false), 180);
  EXPECT_EQ(a1.find_index(360, false), -1);
  EXPECT_EQ(a1.find_index(360, true), 359);
  EXPECT_EQ(a1.size(), 360);
  EXPECT_EQ(a1, a1);
  a1.flip();
  EXPECT_EQ(a1.front(), 359);
  EXPECT_EQ(a1.back(), 0);
  EXPECT_EQ(a1.min_value(), 0);
  EXPECT_EQ(a1.max_value(), 359);
  EXPECT_EQ(a1.coordinate_value(2), 357);
  EXPECT_EQ(a1.find_index(180, false), 179);
  EXPECT_EQ(a1.find_index(360, false), -1);
  EXPECT_EQ(a1.find_index(360, true), 0);
  EXPECT_EQ(a1.size(), 360);
  EXPECT_EQ(a1, a1);
  auto a2 = container::Regular(-180, 179, 360);
  EXPECT_FALSE(a1 == a2);
  EXPECT_FALSE(a1 == container::Undefined());
}
