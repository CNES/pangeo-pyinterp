// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/math/interpolate/bracket_finder.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "pyinterp/eigen.hpp"

namespace pyinterp::math::interpolate {

// =============================================================================
// Helper Functions
// =============================================================================

template <typename T>
auto make_vector(std::initializer_list<T> values) -> Vector<T> {
  Vector<T> v(values.size());
  size_t i = 0;
  for (auto val : values) {
    v[static_cast<int>(i++)] = val;
  }
  return v;
}

// =============================================================================
// Typed test fixture
// =============================================================================

template <typename T>
class BracketFinderTest : public ::testing::Test {
 protected:
  BracketFinder<T> finder_{};
};

using NumericTypes = ::testing::Types<int32_t, int64_t, float, double>;
TYPED_TEST_SUITE(BracketFinderTest, NumericTypes);

// Empty and singleton inputs should not produce a bracket
TYPED_TEST(BracketFinderTest, EmptyAndSingletonNoBracket) {
  // Empty
  {
    auto xa = make_vector<TypeParam>({});
    auto res = this->finder_.search(xa, static_cast<TypeParam>(0));
    EXPECT_FALSE(res.has_value());
  }
  // Singleton (equal and not equal)
  {
    auto xa = make_vector<TypeParam>({static_cast<TypeParam>(5)});
    EXPECT_FALSE(
        this->finder_.search(xa, static_cast<TypeParam>(5)).has_value());
    EXPECT_FALSE(
        this->finder_.search(xa, static_cast<TypeParam>(6)).has_value());
  }
}

// Two elements: endpoints and interior
TYPED_TEST(BracketFinderTest, TwoElementsEndpointsAndBetween) {
  auto xa = make_vector<TypeParam>(
      {static_cast<TypeParam>(0), static_cast<TypeParam>(10)});

  // Endpoints
  {
    auto r0 = this->finder_.search(xa, static_cast<TypeParam>(0));
    ASSERT_TRUE(r0.has_value());
    EXPECT_EQ(r0->first, 0);
    EXPECT_EQ(r0->second, 1);

    auto r1 = this->finder_.search(xa, static_cast<TypeParam>(10));
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->first, 0);
    EXPECT_EQ(r1->second, 1);
  }

  // Interior
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(3));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 0);
    EXPECT_EQ(r->second, 1);
  }

  // Out of range
  {
    EXPECT_FALSE(
        this->finder_.search(xa, static_cast<TypeParam>(-1)).has_value());
    EXPECT_FALSE(
        this->finder_.search(xa, static_cast<TypeParam>(11)).has_value());
  }
}

// Multiple elements: interior points and boundaries
TYPED_TEST(BracketFinderTest, MultipleElementsBetweenPoints) {
  auto xa = make_vector<TypeParam>(
      {static_cast<TypeParam>(0), static_cast<TypeParam>(5),
       static_cast<TypeParam>(10), static_cast<TypeParam>(20)});

  // Between 0 and 5
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(1));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 0);
    EXPECT_EQ(r->second, 1);
  }

  // Exact match on an interior point uses the lower bracket
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(5));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 0);
    EXPECT_EQ(r->second, 1);
  }

  // Between 5 and 10
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(7));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 1);
    EXPECT_EQ(r->second, 2);
  }

  // Exact match on last interior point
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(10));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 1);
    EXPECT_EQ(r->second, 2);
  }

  // Between 10 and 20
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(19));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 2);
    EXPECT_EQ(r->second, 3);
  }

  // Exact match at last element
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(20));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 2);
    EXPECT_EQ(r->second, 3);
  }

  // Out of range
  {
    EXPECT_FALSE(
        this->finder_.search(xa, static_cast<TypeParam>(-2)).has_value());
    EXPECT_FALSE(
        this->finder_.search(xa, static_cast<TypeParam>(25)).has_value());
  }
}

// Duplicates: bracket around the first occurrence
TYPED_TEST(BracketFinderTest, DuplicatesExactMatch) {
  auto xa = make_vector<TypeParam>(
      {static_cast<TypeParam>(0), static_cast<TypeParam>(5),
       static_cast<TypeParam>(5), static_cast<TypeParam>(10)});

  auto r = this->finder_.search(xa, static_cast<TypeParam>(5));
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->first, 0);
  EXPECT_EQ(r->second, 1);
}

// Descending order: two elements
TYPED_TEST(BracketFinderTest, DescendingOrderTwoElements) {
  auto xa = make_vector<TypeParam>(
      {static_cast<TypeParam>(10), static_cast<TypeParam>(0)});

  // Endpoints
  {
    auto r0 = this->finder_.search(xa, static_cast<TypeParam>(10));
    ASSERT_TRUE(r0.has_value());
    EXPECT_EQ(r0->first, 0);
    EXPECT_EQ(r0->second, 1);

    auto r1 = this->finder_.search(xa, static_cast<TypeParam>(0));
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->first, 0);
    EXPECT_EQ(r1->second, 1);
  }

  // Interior
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(5));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 0);
    EXPECT_EQ(r->second, 1);
  }

  // Out of range
  {
    EXPECT_FALSE(
        this->finder_.search(xa, static_cast<TypeParam>(11)).has_value());
    EXPECT_FALSE(
        this->finder_.search(xa, static_cast<TypeParam>(-1)).has_value());
  }
}

// Descending order: multiple elements
TYPED_TEST(BracketFinderTest, DescendingOrderMultipleElements) {
  auto xa = make_vector<TypeParam>(
      {static_cast<TypeParam>(20), static_cast<TypeParam>(10),
       static_cast<TypeParam>(5), static_cast<TypeParam>(0)});

  // Between 20 and 10
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(15));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 0);
    EXPECT_EQ(r->second, 1);
  }

  // Exact match on an interior point
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(10));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 0);
    EXPECT_EQ(r->second, 1);
  }

  // Between 10 and 5
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(7));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 1);
    EXPECT_EQ(r->second, 2);
  }

  // Between 5 and 0
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(2));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 2);
    EXPECT_EQ(r->second, 3);
  }

  // Exact match at last element
  {
    auto r = this->finder_.search(xa, static_cast<TypeParam>(0));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first, 2);
    EXPECT_EQ(r->second, 3);
  }

  // Out of range
  {
    EXPECT_FALSE(
        this->finder_.search(xa, static_cast<TypeParam>(25)).has_value());
    EXPECT_FALSE(
        this->finder_.search(xa, static_cast<TypeParam>(-5)).has_value());
  }
}

}  // namespace pyinterp::math::interpolate
