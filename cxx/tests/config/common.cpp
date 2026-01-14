// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/config/common.hpp"

#include <gtest/gtest.h>

#include <thread>

namespace pyinterp::config {

// ============================================================================
// AxisConfig Tests
// ============================================================================

class AxisConfigTest : public ::testing::Test {};

TEST_F(AxisConfigTest, DefaultConstructor) {
  AxisConfig config;
  // Default should be linear
  EXPECT_EQ(config.method(), AxisMethod::kLinear);
}

TEST_F(AxisConfigTest, ConstructorWithMethod) {
  AxisConfig config(AxisMethod::kNearest);
  EXPECT_EQ(config.method(), AxisMethod::kNearest);
}

TEST_F(AxisConfigTest, LinearFactoryMethod) {
  auto config = AxisConfig::linear();
  EXPECT_EQ(config.method(), AxisMethod::kLinear);
}

TEST_F(AxisConfigTest, NearestFactoryMethod) {
  auto config = AxisConfig::nearest();
  EXPECT_EQ(config.method(), AxisMethod::kNearest);
}

TEST_F(AxisConfigTest, ConstexprUsage) {
  constexpr AxisConfig config = AxisConfig::linear();
  constexpr auto method = config.method();
  EXPECT_EQ(method, AxisMethod::kLinear);
}

// ============================================================================
// ThreadConfig Tests
// ============================================================================

class ThreadConfigTest : public ::testing::Test {};

TEST_F(ThreadConfigTest, DefaultConstructor) {
  ThreadConfig config;
  // Default should be all available threads
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(ThreadConfigTest, WithNumThreadsExplicit) {
  ThreadConfig config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.num_threads(), 4);
  // Original should be unchanged
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(ThreadConfigTest, WithNumThreadsZeroUsesAllThreads) {
  ThreadConfig config;
  auto updated = config.with_num_threads(0);
  EXPECT_EQ(updated.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(ThreadConfigTest, WithNumThreadsChaining) {
  ThreadConfig config;
  auto updated = config.with_num_threads(4).with_num_threads(8);
  EXPECT_EQ(updated.num_threads(), 8);
}

// ============================================================================
// Common Tests
// ============================================================================

class CommonTest : public ::testing::Test {};

TEST_F(CommonTest, DefaultConstructor) {
  Common config;
  EXPECT_FALSE(config.bounds_error());
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(CommonTest, WithBoundsError) {
  Common config;
  auto updated = config.with_bounds_error(true);
  EXPECT_TRUE(updated.bounds_error());
  EXPECT_FALSE(config.bounds_error());  // Original unchanged
}

TEST_F(CommonTest, WithBoundsErrorChaining) {
  Common config;
  auto updated = config.with_bounds_error(true).with_bounds_error(false);
  EXPECT_FALSE(updated.bounds_error());
}

TEST_F(CommonTest, WithNumThreads) {
  Common config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.num_threads(), 4);
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(CommonTest, WithNumThreadsZero) {
  Common config;
  auto updated = config.with_num_threads(0);
  EXPECT_EQ(updated.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(CommonTest, CombinedConfiguration) {
  Common config;
  auto updated = config.with_bounds_error(true).with_num_threads(8);
  EXPECT_TRUE(updated.bounds_error());
  EXPECT_EQ(updated.num_threads(), 8);
}

TEST_F(CommonTest, InheritanceFromThreadConfig) {
  // Common should inherit from ThreadConfig
  Common config;
  ThreadConfig& thread_config = config;
  EXPECT_EQ(thread_config.num_threads(), std::thread::hardware_concurrency());
}

// ============================================================================
// Base Tests
// ============================================================================

// Test helper struct to verify Base class functionality
struct TestFittingModel {
  int dummy_field = 0;
};

struct DerivedConfig : Base<TestFittingModel, DerivedConfig> {
  TestFittingModel spatial_;
  Common common_;
};

class BaseTest : public ::testing::Test {};

TEST_F(BaseTest, WithBoundsError) {
  DerivedConfig config;
  auto updated = config.with_bounds_error(true);
  EXPECT_TRUE(updated.common_.bounds_error());
}

TEST_F(BaseTest, WithNumThreads) {
  DerivedConfig config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.common_.num_threads(), 4);
}

TEST_F(BaseTest, WithSpatial) {
  DerivedConfig config;
  TestFittingModel model{.dummy_field = 42};
  auto updated = config.with_spatial(model);
  EXPECT_EQ(updated.spatial_.dummy_field, model.dummy_field);
  static_assert(std::is_same_v<decltype(updated), DerivedConfig>);
}

TEST_F(BaseTest, MethodChaining) {
  DerivedConfig config;
  auto updated = config.with_bounds_error(true).with_num_threads(8);
  EXPECT_TRUE(updated.common_.bounds_error());
  EXPECT_EQ(updated.common_.num_threads(), 8);
}

}  // namespace pyinterp::config
