// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/math/axis.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp {

// =============================================================================
// Test Fixture
// =============================================================================

template <typename T>
class AxisTestSuite : public testing::Test {
 protected:
  static constexpr T kDefaultEpsilon = static_cast<T>(1e-6);
  static constexpr T kPeriodicCircle = 360;

  void CreateRegularAxis(T start, T stop, T num, bool is_periodic = false) {
    axis_ = std::make_unique<math::Axis<T>>(start, stop, num, kDefaultEpsilon,
                                            is_periodic ? kPeriodicCircle : 0);
  }

  void CreateIrregularAxis(const pyinterp::Vector<T>& values,
                           bool is_periodic = false) {
    axis_ = std::make_unique<math::Axis<T>>(values, kDefaultEpsilon,
                                            is_periodic ? kPeriodicCircle : 0);
  }

  auto axis() -> math::Axis<T>* { return axis_.get(); }

 private:
  std::unique_ptr<math::Axis<T>> axis_{std::make_unique<math::Axis<T>>()};
};

using NumericTypes = testing::Types<int32_t, int64_t, float, double>;
TYPED_TEST_SUITE(AxisTestSuite, NumericTypes);

// =============================================================================
// Helper Functions
// =============================================================================

template <typename T>
void expect_index_pair(
    const std::optional<std::tuple<int64_t, int64_t>>& result,
    int64_t expected_first, int64_t expected_second) {
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(std::get<0>(*result), expected_first);
  EXPECT_EQ(std::get<1>(*result), expected_second);
}

template <typename T>
void expect_index_vector(
    const std::optional<math::axis::IndexWindow>& actual,
    const std::pair<std::vector<int64_t>, std::pair<int64_t, int64_t>>&
        expected) {
  ASSERT_TRUE(actual.has_value());
  auto [indexes, center_indexes] = *actual;
  ASSERT_EQ(indexes.size(), expected.first.size());
  for (size_t i = 0; i < expected.first.size(); ++i) {
    EXPECT_EQ(indexes[i], expected.first[i]) << "Mismatch at index " << i;
  }
  expect_index_pair<T>(std::make_optional(center_indexes),
                       expected.second.first, expected.second.second);
}

// =============================================================================
// Default Constructor Tests
// =============================================================================

TYPED_TEST(AxisTestSuite, DefaultConstructorHasUndefinedValues) {
  auto* axis = this->axis();

  EXPECT_TRUE(math::Fill<TypeParam>::is_fill_value(axis->front()));
  EXPECT_TRUE(math::Fill<TypeParam>::is_fill_value(axis->back()));
  EXPECT_TRUE(math::Fill<TypeParam>::is_fill_value(axis->min_value()));
  EXPECT_TRUE(math::Fill<TypeParam>::is_fill_value(axis->max_value()));
  EXPECT_EQ(axis->size(), 0);
  EXPECT_FALSE(axis->is_periodic());
  EXPECT_TRUE(axis->is_ascending());
  EXPECT_FALSE(axis->is_regular());
  auto repr = std::string(*axis);
  EXPECT_NE(repr.find("Axis(irregular)"), std::string::npos);
}

TYPED_TEST(AxisTestSuite, DefaultConstructorThrowsOnInvalidOperations) {
  auto* axis = this->axis();

  EXPECT_THROW(static_cast<void>(axis->increment()), std::logic_error);
  EXPECT_THROW(static_cast<void>(axis->coordinate_value(0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(axis->slice(0, 1)), std::out_of_range);
}

TYPED_TEST(AxisTestSuite, DefaultConstructorReturnsInvalidIndexes) {
  auto* axis = this->axis();

  EXPECT_EQ(axis->find_index(360, true), -1);
  EXPECT_EQ(axis->find_index(360, false), -1);
  EXPECT_FALSE(axis->find_indexes(360).has_value());
}

// =============================================================================
// Singleton Axis Tests
// =============================================================================

TYPED_TEST(AxisTestSuite, SingletonAxisHasCorrectProperties) {
  this->CreateRegularAxis(0, 1, 1);
  auto* axis = this->axis();

  EXPECT_EQ(axis->front(), 0);
  EXPECT_EQ(axis->back(), 0);
  EXPECT_EQ(axis->min_value(), 0);
  EXPECT_EQ(axis->max_value(), 0);
  EXPECT_EQ(axis->increment(), 1);
  EXPECT_EQ(axis->size(), 1);
  EXPECT_TRUE(axis->is_ascending());
  EXPECT_TRUE(axis->is_regular());
  EXPECT_FALSE(axis->is_periodic());
}

TYPED_TEST(AxisTestSuite, SingletonAxisFindIndexBehavior) {
  this->CreateRegularAxis(0, 1, 1);
  auto* axis = this->axis();

  EXPECT_EQ(axis->find_index(0, false), 0);
  EXPECT_EQ(axis->find_index(1, false), -1);
  EXPECT_EQ(axis->find_index(1, true), 0);
  EXPECT_FALSE(axis->find_indexes(0).has_value());
}

TYPED_TEST(AxisTestSuite, SingletonAxisSliceAndAccess) {
  this->CreateRegularAxis(0, 1, 1);
  auto* axis = this->axis();

  EXPECT_EQ(axis->coordinate_value(0), 0);

  auto slice = axis->slice(0, 1);
  EXPECT_EQ(slice.size(), 1);
  EXPECT_EQ(slice[0], 0);

  EXPECT_THROW(static_cast<void>(axis->coordinate_value(1)), std::exception);
  EXPECT_THROW(static_cast<void>(axis->slice(0, 2)), std::exception);
}

// =============================================================================
// Binary Axis Tests
// =============================================================================

TYPED_TEST(AxisTestSuite, BinaryAxisHasCorrectProperties) {
  this->CreateRegularAxis(0, 1, 2);
  auto* axis = this->axis();

  EXPECT_EQ(axis->front(), 0);
  EXPECT_EQ(axis->back(), 1);
  EXPECT_EQ(axis->min_value(), 0);
  EXPECT_EQ(axis->max_value(), 1);
  EXPECT_EQ(axis->increment(), 1);
  EXPECT_EQ(axis->size(), 2);
  EXPECT_TRUE(axis->is_ascending());
  EXPECT_TRUE(axis->is_regular());
  EXPECT_FALSE(axis->is_periodic());
}

TYPED_TEST(AxisTestSuite, BinaryAxisFindIndexesBoundary) {
  this->CreateRegularAxis(0, 1, 2);
  auto* axis = this->axis();

  expect_index_pair<TypeParam>(axis->find_indexes(0), 0, 1);
  expect_index_pair<TypeParam>(axis->find_indexes(1), 0, 1);
}

TYPED_TEST(AxisTestSuite, BinaryAxisFindIndexesFloat) {
  if constexpr (std::is_floating_point_v<TypeParam>) {
    this->CreateRegularAxis(0, 1, 2);
    auto* axis = this->axis();

    EXPECT_FALSE(axis->find_indexes(static_cast<TypeParam>(-0.1)));
    EXPECT_FALSE(axis->find_indexes(static_cast<TypeParam>(1.1)));
    expect_index_pair<TypeParam>(
        axis->find_indexes(static_cast<TypeParam>(0.4)), 0, 1);
    expect_index_pair<TypeParam>(
        axis->find_indexes(static_cast<TypeParam>(0.6)), 0, 1);
  }
}

TYPED_TEST(AxisTestSuite, BinaryAxisFindIndexesInteger) {
  if constexpr (std::is_integral_v<TypeParam>) {
    this->CreateRegularAxis(0, 1, 2);
    auto* axis = this->axis();

    EXPECT_FALSE(axis->find_indexes(-1));
    EXPECT_FALSE(axis->find_indexes(2));
    expect_index_pair<TypeParam>(axis->find_indexes(0), 0, 1);
  }
}

// =============================================================================
// Periodic Axis Tests (Longitude Wrapping)
// =============================================================================

TYPED_TEST(AxisTestSuite, PeriodicAxis0To359Properties) {
  this->CreateRegularAxis(0, 359, 360, true);
  auto* axis = this->axis();

  EXPECT_EQ(axis->front(), 0);
  EXPECT_EQ(axis->back(), 359);
  EXPECT_EQ(axis->min_value(), 0);
  EXPECT_EQ(axis->max_value(), 359);
  EXPECT_EQ(axis->increment(), 1);
  EXPECT_EQ(axis->size(), 360);
  EXPECT_TRUE(axis->is_ascending());
  EXPECT_TRUE(axis->is_regular());
  EXPECT_TRUE(axis->is_periodic());
  auto repr = std::string(*axis);
  EXPECT_EQ(repr, R"(Axis(regular, period=360)
  range: [0, 359]
  step: 1
  size: 360)");
}

TYPED_TEST(AxisTestSuite, PeriodicAxis0To359WrapBehavior) {
  this->CreateRegularAxis(0, 359, 360, true);
  auto* axis = this->axis();

  EXPECT_EQ(axis->find_index(0, false), 0);
  EXPECT_EQ(axis->find_index(360, true), 0);
  EXPECT_EQ(axis->find_index(360, false), 0);

  expect_index_pair<TypeParam>(axis->find_indexes(360), 0, 1);
  expect_index_pair<TypeParam>(axis->find_indexes(370), 10, 11);
}

TYPED_TEST(AxisTestSuite, PeriodicAxis0To359NegativeWrap) {
  if constexpr (std::is_floating_point_v<TypeParam>) {
    this->CreateRegularAxis(0, 359, 360, true);
    auto* axis = this->axis();

    expect_index_pair<TypeParam>(
        axis->find_indexes(static_cast<TypeParam>(-9.5)), 350, 351);
  } else {
    this->CreateRegularAxis(0, 359, 360, true);
    auto* axis = this->axis();

    expect_index_pair<TypeParam>(axis->find_indexes(-10), 350, 351);
  }
}

TYPED_TEST(AxisTestSuite, PeriodicAxisFlippedBehavior) {
  this->CreateRegularAxis(0, 359, 360, true);
  auto* axis = this->axis();
  axis->flip();

  EXPECT_EQ(axis->front(), 359);
  EXPECT_EQ(axis->back(), 0);
  EXPECT_EQ(axis->increment(), -1);
  EXPECT_FALSE(axis->is_ascending());
  EXPECT_TRUE(axis->is_periodic());
  EXPECT_TRUE(axis->is_regular());
}

TYPED_TEST(AxisTestSuite, PeriodicAxisMinus180To179Properties) {
  auto axis = math::Axis<TypeParam>(-180, 179, 360, this->kDefaultEpsilon,
                                    this->kPeriodicCircle);

  EXPECT_EQ(axis.front(), -180);
  EXPECT_EQ(axis.back(), 179);
  EXPECT_EQ(axis.min_value(), -180);
  EXPECT_EQ(axis.max_value(), 179);
  EXPECT_EQ(axis.increment(), 1);
  EXPECT_TRUE(axis.is_ascending());
  EXPECT_TRUE(axis.is_periodic());
  EXPECT_TRUE(axis.is_regular());
}

// =============================================================================
// Irregular Axis Tests
// =============================================================================

TEST(Axis, IrregularAxisProperties) {
  std::vector<double> values{-89.0,      -88.908818, -88.809323, -88.700757,
                             -88.582294, -88.453032, -88.311987, -88.158087,
                             -87.990161, -87.806932};

  math::Axis<double> axis(
      Eigen::Map<Eigen::VectorXd>(values.data(), values.size()), 1e-6);

  EXPECT_EQ(axis.front(), -89.0);
  EXPECT_EQ(axis.back(), -87.806932);
  EXPECT_EQ(axis.min_value(), -89.0);
  EXPECT_EQ(axis.max_value(), -87.806932);
  EXPECT_EQ(axis.size(), values.size());
  EXPECT_FALSE(axis.is_periodic());
  EXPECT_TRUE(axis.is_ascending());
  EXPECT_FALSE(axis.is_regular());
  EXPECT_THROW(static_cast<void>(axis.increment()), std::logic_error);
  auto repr = std::string(axis);
  EXPECT_EQ(repr,
            R"(Axis(irregular)
  values: [-89, -88.908818, -88.809323, ..., -88.158087, -87.990161, -87.806932]
  size: 10)");
}

TEST(Axis, IrregularAxisFindIndex) {
  std::vector<double> values{-89.0, -50.0, 0.0, 50.0, 88.940374};

  math::Axis<double> axis(
      Eigen::Map<Eigen::VectorXd>(values.data(), values.size()), 1e-6);

  EXPECT_EQ(axis.find_index(0.0, false), 2);
  EXPECT_EQ(axis.find_index(-90.0, false), -1);
  EXPECT_EQ(axis.find_index(-90.0, true), 0);
  EXPECT_EQ(axis.find_index(90.0, false), -1);
  EXPECT_EQ(axis.find_index(90.0, true), 4);
}

// =============================================================================
// Constant Values Tests
// =============================================================================

TYPED_TEST(AxisTestSuite, ConstantValuesThrowsOnDuplicates) {
  auto values = pyinterp::Vector<TypeParam>(5);
  values[0] = 0;
  values[1] = 1;
  values[2] = 5;
  values[3] = 5;  // Duplicate
  values[4] = 5;  // Duplicate

  EXPECT_THROW(this->CreateIrregularAxis(values), std::invalid_argument);
}

TYPED_TEST(AxisTestSuite, ConstantValuesThrowsOnAllSame) {
  auto values = pyinterp::Vector<TypeParam>(5);
  values[0] = 5;
  values[1] = 5;
  values[2] = 5;
  values[3] = 5;
  values[4] = 5;

  EXPECT_THROW(this->CreateIrregularAxis(values), std::invalid_argument);
}

// =============================================================================
// Search Window Tests
// =============================================================================

TYPED_TEST(AxisTestSuite, SearchWindowPeriodicAxis) {
  this->CreateRegularAxis(-180, 179, 360, true);
  auto* axis = this->axis();

  auto indexes = axis->find_indexes(0, 5, math::axis::kUndef);
  expect_index_vector<TypeParam>(
      indexes, {{176, 177, 178, 179, 180, 181, 182, 183, 184, 185}, {4, 5}});

  indexes = axis->find_indexes(-180, 5, math::axis::kUndef);
  expect_index_vector<TypeParam>(
      indexes, {{356, 357, 358, 359, 0, 1, 2, 3, 4, 5}, {4, 5}});
}

TYPED_TEST(AxisTestSuite, SearchWindowBoundaryModes) {
  this->CreateRegularAxis(0, 9, 10, false);
  auto* axis = this->axis();

  // Symmetric mode
  auto indexes = axis->find_indexes(1, 4, math::axis::kSym);
  expect_index_vector<TypeParam>(indexes, {{2, 1, 0, 1, 2, 3, 4, 5}, {3, 4}});

  // Wrap mode
  indexes = axis->find_indexes(1, 4, math::axis::kWrap);
  expect_index_vector<TypeParam>(indexes, {{8, 9, 0, 1, 2, 3, 4, 5}, {3, 4}});
  // Expand mode
  indexes = axis->find_indexes(1, 4, math::axis::kExpand);
  expect_index_vector<TypeParam>(indexes, {{0, 0, 0, 1, 2, 3, 4, 5}, {3, 4}});

  // Shrink mode
  indexes = axis->find_indexes(1, 4, math::axis::kShrink);
  expect_index_vector<TypeParam>(indexes, {{0, 1, 2, 3, 4, 5}, {1, 2}});
}

// =============================================================================
// Timestamp Tests
// =============================================================================

TEST(Axis, TimestampFindIndex) {
  auto axis = math::Axis<int64_t>(946684800, 946771140, 1440, 0);

  EXPECT_EQ(axis.find_index(946684880, true), 1);
  EXPECT_EQ(axis.find_index(946684900, true), 2);

  axis.flip();
  EXPECT_EQ(axis.find_index(946684880, true), 1438);
  EXPECT_EQ(axis.find_index(946684900, true), 1437);
}

// =============================================================================
// Nearest Index Tests
// =============================================================================

TYPED_TEST(AxisTestSuite, FindNearestIndexPeriodicAxis) {
  this->CreateRegularAxis(0, 355, 72, true);
  auto* axis = this->axis();

  EXPECT_EQ(axis->find_index(356, false), 71);
  EXPECT_EQ(axis->find_index(358, false), 0);
  EXPECT_EQ(axis->find_index(-2, false), 0);
  EXPECT_EQ(axis->find_index(-4, false), 71);
}

TYPED_TEST(AxisTestSuite, FindNearestIndexCenteredPeriodicAxis) {
  this->CreateRegularAxis(-180, 175, 72, true);
  auto* axis = this->axis();

  EXPECT_EQ(axis->find_index(176, false), 71);
  EXPECT_EQ(axis->find_index(178, false), 0);
  EXPECT_EQ(axis->find_index(-182, false), 0);
  EXPECT_EQ(axis->find_index(-184, false), 71);
}

// =============================================================================
// pack/unpack Tests
// =============================================================================
TYPED_TEST(AxisTestSuite, RegularSerializationDeserialization) {
  this->CreateRegularAxis(0, 359, 360, true);
  auto* axis = this->axis();

  auto state_reader = serialization::Reader(axis->pack());
  auto restored_axis = math::Axis<TypeParam>::unpack(state_reader);

  EXPECT_EQ(*axis, restored_axis);
}

TYPED_TEST(AxisTestSuite, IrregularSerializationDeserialization) {
  pyinterp::Vector<TypeParam> values(10);
  for (int64_t ix = 0; ix < 10; ++ix) {
    values[ix] = TypeParam(ix * ix);
  }
  this->CreateIrregularAxis(values);

  auto* axis = this->axis();

  auto state_reader = serialization::Reader(axis->pack());
  auto restored_axis = math::Axis<TypeParam>::unpack(state_reader);

  EXPECT_EQ(*axis, restored_axis);
}

TYPED_TEST(AxisTestSuite, UndefinedSerializationDeserialization) {
  auto axis = math::Axis<TypeParam>();

  auto state_reader = serialization::Reader(axis.pack());
  auto restored_axis = math::Axis<TypeParam>::unpack(state_reader);

  EXPECT_EQ(axis, restored_axis);
}

}  // namespace pyinterp
