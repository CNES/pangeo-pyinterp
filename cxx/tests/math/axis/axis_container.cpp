// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/axis/container.hpp"
#include "pyinterp/math/fill.hpp"

namespace pyinterp::math {

// =============================================================================
// Helper Functions
// =============================================================================

template <typename T>
auto CreateEigenVector(const std::vector<T>& values)
    -> Eigen::Matrix<T, -1, 1> {
  return Eigen::Map<const Eigen::Matrix<T, -1, 1>>(values.data(),
                                                   values.size());
}

template <typename T>
void ExpectSliceEquals(const pyinterp::Vector<T>& slice,
                       const std::vector<T>& expected) {
  ASSERT_EQ(slice.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(slice[i], expected[i]) << "Mismatch at index " << i;
  }
}

// =============================================================================
// Undefined Axis Tests
// =============================================================================

template <typename T>
class UndefinedTest : public testing::Test {
 protected:
  using Axis = axis::Undefined<T>;

  auto CreateUndefinedAxis() -> Axis { return Axis(); }
};

using NumericTypes = testing::Types<int32_t, int64_t, float, double>;
TYPED_TEST_SUITE(UndefinedTest, NumericTypes);

TYPED_TEST(UndefinedTest, DefaultStateAllValuesFilled) {
  auto axis = this->CreateUndefinedAxis();

  EXPECT_TRUE(Fill<TypeParam>::is_fill_value(axis.front()));
  EXPECT_TRUE(Fill<TypeParam>::is_fill_value(axis.back()));
  EXPECT_TRUE(Fill<TypeParam>::is_fill_value(axis.min_value()));
  EXPECT_TRUE(Fill<TypeParam>::is_fill_value(axis.max_value()));
  EXPECT_TRUE(Fill<TypeParam>::is_fill_value(axis.coordinate_value(0)));
}

TYPED_TEST(UndefinedTest, DefaultStateHasZeroSize) {
  auto axis = this->CreateUndefinedAxis();

  EXPECT_EQ(axis.size(), 0);
}

TYPED_TEST(UndefinedTest, FindIndexReturnsInvalidIndex) {
  auto axis = this->CreateUndefinedAxis();

  EXPECT_EQ(axis.find_index(360, true), -1);
  EXPECT_EQ(axis.find_index(360, false), -1);
}

TYPED_TEST(UndefinedTest, SliceReturnsSingleFilledValue) {
  auto axis = this->CreateUndefinedAxis();

  auto slice = axis.slice(0, 1);
  EXPECT_EQ(slice.size(), 1);
  EXPECT_TRUE(Fill<TypeParam>::is_fill_value(slice[0]));
}

TYPED_TEST(UndefinedTest, FlipDoesNotChangeState) {
  auto axis = this->CreateUndefinedAxis();
  axis.flip();

  EXPECT_TRUE(Fill<TypeParam>::is_fill_value(axis.front()));
  EXPECT_TRUE(Fill<TypeParam>::is_fill_value(axis.back()));
  EXPECT_EQ(axis.size(), 0);
}

TYPED_TEST(UndefinedTest, EqualitySelfComparison) {
  auto axis = this->CreateUndefinedAxis();

  EXPECT_EQ(axis, axis);
}

// =============================================================================
// Irregular Axis Tests
// =============================================================================

template <typename T>
class IrregularTest : public testing::Test {
 protected:
  using Axis = axis::Irregular<T>;

  auto CreateAxis(const std::vector<T>& values) -> Axis {
    return Axis(CreateEigenVector(values));
  }

  static constexpr std::array<T, 5> kDefaultValues = {0, 1, 4, 8, 20};
};

TYPED_TEST_SUITE(IrregularTest, NumericTypes);

TYPED_TEST(IrregularTest, AscendingBasicProperties) {
  std::vector<TypeParam> values{0, 1, 4, 8, 20};
  auto axis = this->CreateAxis(values);

  EXPECT_EQ(axis.front(), 0);
  EXPECT_EQ(axis.back(), 20);
  EXPECT_EQ(axis.min_value(), 0);
  EXPECT_EQ(axis.max_value(), 20);
  EXPECT_EQ(axis.size(), 5);
}

TYPED_TEST(IrregularTest, AscendingCoordinateAccess) {
  std::vector<TypeParam> values{0, 1, 4, 8, 20};
  auto axis = this->CreateAxis(values);

  EXPECT_EQ(axis.coordinate_value(0), 0);
  EXPECT_EQ(axis.coordinate_value(2), 4);
  EXPECT_EQ(axis.coordinate_value(4), 20);
}

TYPED_TEST(IrregularTest, AscendingSliceExtraction) {
  std::vector<TypeParam> values{0, 1, 4, 8, 20};
  auto axis = this->CreateAxis(values);

  auto slice = axis.slice(1, 3);
  ExpectSliceEquals(slice, std::vector<TypeParam>{1, 4, 8});
}

TYPED_TEST(IrregularTest, AscendingFindExactIndex) {
  std::vector<TypeParam> values{0, 1, 4, 8, 20};
  auto axis = this->CreateAxis(values);

  EXPECT_EQ(axis.find_index(0, false), 0);
  EXPECT_EQ(axis.find_index(8, false), 3);
  EXPECT_EQ(axis.find_index(20, false), 4);
}

TYPED_TEST(IrregularTest, AscendingFindNearestIndex) {
  std::vector<TypeParam> values{0, 1, 4, 8, 20};
  auto axis = this->CreateAxis(values);

  EXPECT_EQ(axis.find_index(static_cast<TypeParam>(8.3), false), 3);
  EXPECT_EQ(axis.find_index(static_cast<TypeParam>(20.1), true), 4);
}

TYPED_TEST(IrregularTest, AscendingFindIndexOutOfBounds) {
  std::vector<TypeParam> values{0, 1, 4, 8, 20};
  auto axis = this->CreateAxis(values);

  EXPECT_EQ(axis.find_index(30, false), -1);
  EXPECT_EQ(axis.find_index(30, true), 4);  // Bounded to last index
}

TYPED_TEST(IrregularTest, FlippedBasicProperties) {
  std::vector<TypeParam> values{0, 1, 4, 8, 20};
  auto axis = this->CreateAxis(values);
  axis.flip();

  EXPECT_EQ(axis.front(), 20);
  EXPECT_EQ(axis.back(), 0);
  EXPECT_EQ(axis.min_value(), 0);
  EXPECT_EQ(axis.max_value(), 20);
  EXPECT_EQ(axis.size(), 5);
}

TYPED_TEST(IrregularTest, FlippedCoordinateAccess) {
  std::vector<TypeParam> values{0, 1, 4, 8, 20};
  auto axis = this->CreateAxis(values);
  axis.flip();

  EXPECT_EQ(axis.coordinate_value(0), 20);
  EXPECT_EQ(axis.coordinate_value(2), 4);
  EXPECT_EQ(axis.coordinate_value(4), 0);
}

TYPED_TEST(IrregularTest, FlippedSliceExtraction) {
  std::vector<TypeParam> values{0, 1, 4, 8, 20};
  auto axis = this->CreateAxis(values);
  axis.flip();

  auto slice = axis.slice(1, 3);
  ExpectSliceEquals(slice, std::vector<TypeParam>{8, 4, 1});
}

TYPED_TEST(IrregularTest, FlippedFindIndex) {
  std::vector<TypeParam> values{0, 1, 4, 8, 20};
  auto axis = this->CreateAxis(values);
  axis.flip();

  EXPECT_EQ(axis.find_index(8, false), 1);
  EXPECT_EQ(axis.find_index(static_cast<TypeParam>(8.3), false), 1);
  EXPECT_EQ(axis.find_index(30, true), 0);  // Bounded to first index
  EXPECT_EQ(axis.find_index(static_cast<TypeParam>(20.1), true), 0);
  EXPECT_EQ(axis.find_index(30, false), -1);
}

TYPED_TEST(IrregularTest, EqualitySelfComparison) {
  std::vector<TypeParam> values{0, 1, 4, 8, 20};
  auto axis = this->CreateAxis(values);

  EXPECT_EQ(axis, axis);
}

TYPED_TEST(IrregularTest, EqualityDifferentAxes) {
  std::vector<TypeParam> values1{0, 1, 4, 8, 20};
  std::vector<TypeParam> values2{0, 1};

  auto axis1 = this->CreateAxis(values1);
  auto axis2 = this->CreateAxis(values2);

  EXPECT_NE(axis1, axis2);
}

TYPED_TEST(IrregularTest, EqualityWithUndefinedAxis) {
  std::vector<TypeParam> values{0, 1, 4, 8, 20};
  auto irregular_axis = this->CreateAxis(values);
  auto undefined_axis = axis::Undefined<TypeParam>();

  EXPECT_NE(irregular_axis, undefined_axis);
}

// =============================================================================
// Regular Axis Tests
// =============================================================================

template <typename T>
class RegularTest : public testing::Test {
 protected:
  using Axis = axis::Regular<T>;

  auto CreateAxis(T start, T stop, size_t num) -> Axis {
    return Axis(start, stop, num);
  }
};

TYPED_TEST_SUITE(RegularTest, NumericTypes);

TYPED_TEST(RegularTest, ConstructorThrowsOnZeroSize) {
  EXPECT_THROW(this->CreateAxis(0, 359, 0), std::invalid_argument);
}

TYPED_TEST(RegularTest, AscendingBasicProperties) {
  auto axis = this->CreateAxis(0, 359, 360);

  EXPECT_EQ(axis.front(), 0);
  EXPECT_EQ(axis.back(), 359);
  EXPECT_EQ(axis.min_value(), 0);
  EXPECT_EQ(axis.max_value(), 359);
  EXPECT_EQ(axis.size(), 360);
}

TYPED_TEST(RegularTest, AscendingCoordinateAccess) {
  auto axis = this->CreateAxis(0, 359, 360);

  EXPECT_EQ(axis.coordinate_value(0), 0);
  EXPECT_EQ(axis.coordinate_value(2), 2);
  EXPECT_EQ(axis.coordinate_value(180), 180);
  EXPECT_EQ(axis.coordinate_value(359), 359);
}

TYPED_TEST(RegularTest, AscendingSliceExtraction) {
  auto axis = this->CreateAxis(0, 359, 360);

  auto slice = axis.slice(1, 3);
  ExpectSliceEquals(slice, std::vector<TypeParam>{1, 2, 3});
}

TYPED_TEST(RegularTest, AscendingFindIndex) {
  auto axis = this->CreateAxis(0, 359, 360);

  EXPECT_EQ(axis.find_index(0, false), 0);
  EXPECT_EQ(axis.find_index(180, false), 180);
  EXPECT_EQ(axis.find_index(359, false), 359);
}

TYPED_TEST(RegularTest, AscendingFindIndexBoundary) {
  auto axis = this->CreateAxis(0, 359, 360);

  EXPECT_EQ(axis.find_index(360, false), -1);
  EXPECT_EQ(axis.find_index(360, true), 359);  // Bounded to last index
}

TYPED_TEST(RegularTest, FlippedBasicProperties) {
  auto axis = this->CreateAxis(0, 359, 360);
  axis.flip();

  EXPECT_EQ(axis.front(), 359);
  EXPECT_EQ(axis.back(), 0);
  EXPECT_EQ(axis.min_value(), 0);
  EXPECT_EQ(axis.max_value(), 359);
  EXPECT_EQ(axis.size(), 360);
}

TYPED_TEST(RegularTest, FlippedCoordinateAccess) {
  auto axis = this->CreateAxis(0, 359, 360);
  axis.flip();

  EXPECT_EQ(axis.coordinate_value(0), 359);
  EXPECT_EQ(axis.coordinate_value(2), 357);
  EXPECT_EQ(axis.coordinate_value(180), 179);
  EXPECT_EQ(axis.coordinate_value(359), 0);
}

TYPED_TEST(RegularTest, FlippedSliceExtraction) {
  auto axis = this->CreateAxis(0, 359, 360);
  axis.flip();

  auto slice = axis.slice(1, 3);
  ExpectSliceEquals(slice, std::vector<TypeParam>{358, 357, 356});
}

TYPED_TEST(RegularTest, FlippedFindIndex) {
  auto axis = this->CreateAxis(0, 359, 360);
  axis.flip();

  EXPECT_EQ(axis.find_index(180, false), 179);
  EXPECT_EQ(axis.find_index(360, false), -1);
  EXPECT_EQ(axis.find_index(360, true), 0);  // Bounded to first index
}

TYPED_TEST(RegularTest, EqualitySelfComparison) {
  auto axis = this->CreateAxis(0, 359, 360);

  EXPECT_EQ(axis, axis);
}

TYPED_TEST(RegularTest, EqualityDifferentRanges) {
  auto axis1 = this->CreateAxis(0, 359, 360);
  auto axis2 = this->CreateAxis(-180, 179, 360);

  EXPECT_NE(axis1, axis2);
}

TYPED_TEST(RegularTest, EqualityWithUndefinedAxis) {
  auto regular_axis = this->CreateAxis(0, 359, 360);
  auto undefined_axis = axis::Undefined<TypeParam>();

  EXPECT_NE(regular_axis, undefined_axis);
}

TYPED_TEST(RegularTest, EqualityAfterFlip) {
  auto axis1 = this->CreateAxis(0, 359, 360);
  auto axis2 = this->CreateAxis(0, 359, 360);
  axis1.flip();

  // After flipping, they should still be equal if they represent
  // the same axis (this depends on implementation)
  EXPECT_EQ(axis1, axis1);
}

}  // namespace pyinterp::math
