// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include "pyinterp/detail/axis/container.hpp"

namespace container = pyinterp::detail::axis::container;
namespace math = pyinterp::detail::math;

using Implementations = testing::Types<int32_t, int64_t, float, double>;

template <typename T>
class UndefinedTest : public testing::Test {
 public:
  using Axis = container::Undefined<T>;
};
TYPED_TEST_SUITE(UndefinedTest, Implementations);

TYPED_TEST(UndefinedTest, undefined) {
  // undefined axis
  auto a1 = typename TestFixture::Axis();
  a1.flip();
  EXPECT_TRUE(math::Fill<TypeParam>::is(a1.front()));
  EXPECT_TRUE(math::Fill<TypeParam>::is(a1.back()));
  EXPECT_TRUE(math::Fill<TypeParam>::is(a1.min_value()));
  EXPECT_TRUE(math::Fill<TypeParam>::is(a1.max_value()));
  EXPECT_TRUE(math::Fill<TypeParam>::is(a1.coordinate_value(0)));
  auto slice = a1.slice(0, 1);
  EXPECT_TRUE(slice.size() == 1);
  EXPECT_TRUE(math::Fill<TypeParam>::is(slice[0]));

  EXPECT_EQ(a1.size(), 0);
  EXPECT_EQ(a1.find_index(360, true), -1);
  EXPECT_EQ(a1.find_index(360, false), -1);
  EXPECT_EQ(a1, a1);
}

template <typename T>
class IrregularTest : public testing::Test {
 public:
  using Axis = container::Irregular<T>;
};
TYPED_TEST_SUITE(IrregularTest, Implementations);

TYPED_TEST(IrregularTest, irregular) {
  // irregular axis
  auto values = std::vector<TypeParam>{0, 1, 4, 8, 20};
  auto a1 =
      typename TestFixture::Axis(Eigen::Map<Eigen::Matrix<TypeParam, -1, 1>>(
          values.data(), values.size()));
  EXPECT_EQ(a1.front(), 0);
  EXPECT_EQ(a1.back(), 20);
  EXPECT_EQ(a1.min_value(), 0);
  EXPECT_EQ(a1.max_value(), 20);
  EXPECT_EQ(a1.coordinate_value(2), 4);
  auto slice = a1.slice(1, 3);
  EXPECT_EQ(slice.size(), 3);
  EXPECT_EQ(slice[0], 1);
  EXPECT_EQ(slice[1], 4);
  EXPECT_EQ(slice[2], 8);
  EXPECT_EQ(a1.find_index(8, false), 3);
  EXPECT_EQ(a1.find_index(static_cast<TypeParam>(8.3), false), 3);
  EXPECT_EQ(a1.find_index(30, true), 4);
  EXPECT_EQ(a1.find_index(static_cast<TypeParam>(20.1), true), 4);
  EXPECT_EQ(a1.find_index(30, false), -1);
  EXPECT_EQ(a1.size(), 5);
  EXPECT_EQ(a1, a1);
  a1.flip();
  EXPECT_EQ(a1.front(), 20);
  EXPECT_EQ(a1.back(), 0);
  EXPECT_EQ(a1.min_value(), 0);
  EXPECT_EQ(a1.max_value(), 20);
  EXPECT_EQ(a1.coordinate_value(2), 4);
  slice = a1.slice(1, 3);
  EXPECT_EQ(slice.size(), 3);
  EXPECT_EQ(slice[0], 8);
  EXPECT_EQ(slice[1], 4);
  EXPECT_EQ(slice[2], 1);
  EXPECT_EQ(a1.find_index(8, false), 1);
  EXPECT_EQ(a1.find_index(static_cast<TypeParam>(8.3), false), 1);
  EXPECT_EQ(a1.find_index(30, true), 0);
  EXPECT_EQ(a1.find_index(static_cast<TypeParam>(20.1), true), 0);
  EXPECT_EQ(a1.find_index(30, false), -1);
  EXPECT_EQ(a1.size(), 5);
  EXPECT_EQ(a1, a1);
  values = std::vector<TypeParam>{0, 1};
  auto a2 =
      typename TestFixture::Axis(Eigen::Map<Eigen::Matrix<TypeParam, -1, 1>>(
          values.data(), values.size()));
  EXPECT_FALSE(a1 == a2);
  EXPECT_FALSE(a1 == container::Undefined<TypeParam>());
}

template <typename T>
class RegularTest : public testing::Test {
 public:
  using Axis = container::Regular<T>;
};
TYPED_TEST_SUITE(RegularTest, Implementations);

TYPED_TEST(RegularTest, irregular) {
  // regular axis
  EXPECT_THROW(typename TestFixture::Axis(0, 359, 0), std::invalid_argument);
  auto a1 = typename TestFixture::Axis(0, 359, 360);
  EXPECT_EQ(a1.front(), 0);
  EXPECT_EQ(a1.back(), 359);
  EXPECT_EQ(a1.min_value(), 0);
  EXPECT_EQ(a1.max_value(), 359);
  EXPECT_EQ(a1.coordinate_value(2), 2);
  auto slice = a1.slice(1, 3);
  EXPECT_EQ(slice.size(), 3);
  EXPECT_EQ(slice[0], 1);
  EXPECT_EQ(slice[1], 2);
  EXPECT_EQ(slice[2], 3);
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
  slice = a1.slice(1, 3);
  EXPECT_EQ(slice.size(), 3);
  EXPECT_EQ(slice[0], 358);
  EXPECT_EQ(slice[1], 357);
  EXPECT_EQ(slice[2], 356);
  EXPECT_EQ(a1.find_index(180, false), 179);
  EXPECT_EQ(a1.find_index(360, false), -1);
  EXPECT_EQ(a1.find_index(360, true), 0);
  EXPECT_EQ(a1.size(), 360);
  EXPECT_EQ(a1, a1);
  auto a2 = typename TestFixture::Axis(-180, 179, 360);
  EXPECT_FALSE(a1 == a2);
  EXPECT_FALSE(a1 == container::Undefined<TypeParam>());
}
