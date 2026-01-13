// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/config/fill.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <thread>

namespace pyinterp::config::fill {

// ============================================================================
// FirstGuess Tests
// ============================================================================

class FirstGuessTest : public ::testing::Test {};

TEST_F(FirstGuessTest, ParseFirstGuessZonalAverage) {
  auto strategy = parse_first_guess("zonal_average");
  EXPECT_EQ(strategy, FirstGuess::kZonalAverage);
}

TEST_F(FirstGuessTest, ParseFirstGuessZero) {
  auto strategy = parse_first_guess("zero");
  EXPECT_EQ(strategy, FirstGuess::kZero);
}

TEST_F(FirstGuessTest, ParseFirstGuessUnknownThrows) {
  EXPECT_THROW(static_cast<void>(parse_first_guess("invalid")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(parse_first_guess("")), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(parse_first_guess("Zero")),
               std::invalid_argument);  // case-sensitive
  EXPECT_THROW(static_cast<void>(parse_first_guess("ZONAL_AVERAGE")),
               std::invalid_argument);
}

// ============================================================================
// LoessValueType Tests
// ============================================================================

class LoessValueTypeTest : public ::testing::Test {};

TEST_F(LoessValueTypeTest, ParseLoessValueTypeUndefined) {
  auto value_type = parse_loess_value_type("undefined");
  EXPECT_EQ(value_type, LoessValueType::kUndefined);
}

TEST_F(LoessValueTypeTest, ParseLoessValueTypeDefined) {
  auto value_type = parse_loess_value_type("defined");
  EXPECT_EQ(value_type, LoessValueType::kDefined);
}

TEST_F(LoessValueTypeTest, ParseLoessValueTypeAll) {
  auto value_type = parse_loess_value_type("all");
  EXPECT_EQ(value_type, LoessValueType::kAll);
}

TEST_F(LoessValueTypeTest, ParseLoessValueTypeUnknownThrows) {
  EXPECT_THROW(static_cast<void>(parse_loess_value_type("invalid")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(parse_loess_value_type("")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(parse_loess_value_type("Undefined")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(parse_loess_value_type("ALL")),
               std::invalid_argument);
}

// ============================================================================
// Loess Config Tests
// ============================================================================

class LoessTest : public ::testing::Test {};

TEST_F(LoessTest, DefaultConstructor) {
  Loess config;
  EXPECT_EQ(config.first_guess(), FirstGuess::kZonalAverage);
  EXPECT_TRUE(config.is_periodic());
  EXPECT_EQ(config.max_iterations(), 500);
  EXPECT_DOUBLE_EQ(config.epsilon(), 1e-4);
  EXPECT_EQ(config.value_type(), LoessValueType::kUndefined);
  EXPECT_EQ(config.nx(), 3);
  EXPECT_EQ(config.ny(), 3);
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(LoessTest, WithFirstGuess) {
  Loess config;
  auto updated = config.with_first_guess(FirstGuess::kZero);
  EXPECT_EQ(updated.first_guess(), FirstGuess::kZero);
  EXPECT_EQ(config.first_guess(),
            FirstGuess::kZonalAverage);  // Original unchanged
}

TEST_F(LoessTest, WithIsPeriodic) {
  Loess config;
  auto updated = config.with_is_periodic(false);
  EXPECT_FALSE(updated.is_periodic());
  EXPECT_TRUE(config.is_periodic());
}

TEST_F(LoessTest, WithMaxIterations) {
  Loess config;
  auto updated = config.with_max_iterations(1000);
  EXPECT_EQ(updated.max_iterations(), 1000);
  EXPECT_EQ(config.max_iterations(), 500);
}

TEST_F(LoessTest, WithEpsilon) {
  Loess config;
  auto updated = config.with_epsilon(1e-6);
  EXPECT_DOUBLE_EQ(updated.epsilon(), 1e-6);
  EXPECT_DOUBLE_EQ(config.epsilon(), 1e-4);
}

TEST_F(LoessTest, WithValueType) {
  Loess config;
  auto updated = config.with_value_type(LoessValueType::kAll);
  EXPECT_EQ(updated.value_type(), LoessValueType::kAll);
  EXPECT_EQ(config.value_type(), LoessValueType::kUndefined);
}

TEST_F(LoessTest, WithNx) {
  Loess config;
  auto updated = config.with_nx(5);
  EXPECT_EQ(updated.nx(), 5);
  EXPECT_EQ(config.nx(), 3);
}

TEST_F(LoessTest, WithNy) {
  Loess config;
  auto updated = config.with_ny(7);
  EXPECT_EQ(updated.ny(), 7);
  EXPECT_EQ(config.ny(), 3);
}

TEST_F(LoessTest, WithNxZeroThrows) {
  Loess config;
  volatile uint32_t zero = 0;  // to prevent compiler optimization
  EXPECT_THROW(static_cast<void>(config.with_nx(zero)), std::invalid_argument);
}

TEST_F(LoessTest, WithNyZeroThrows) {
  Loess config;
  volatile uint32_t zero = 0;  // to prevent compiler optimization
  EXPECT_THROW(static_cast<void>(config.with_ny(zero)), std::invalid_argument);
}

TEST_F(LoessTest, WithNumThreads) {
  Loess config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.num_threads(), 4);
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(LoessTest, MethodChaining) {
  Loess config;
  auto updated = config.with_first_guess(FirstGuess::kZero)
                     .with_is_periodic(false)
                     .with_max_iterations(1000)
                     .with_epsilon(1e-6)
                     .with_value_type(LoessValueType::kAll)
                     .with_nx(5)
                     .with_ny(7)
                     .with_num_threads(8);

  EXPECT_EQ(updated.first_guess(), FirstGuess::kZero);
  EXPECT_FALSE(updated.is_periodic());
  EXPECT_EQ(updated.max_iterations(), 1000);
  EXPECT_DOUBLE_EQ(updated.epsilon(), 1e-6);
  EXPECT_EQ(updated.value_type(), LoessValueType::kAll);
  EXPECT_EQ(updated.nx(), 5);
  EXPECT_EQ(updated.ny(), 7);
  EXPECT_EQ(updated.num_threads(), 8);
}

// ============================================================================
// FFTInpaint Config Tests
// ============================================================================

class FFTInpaintTest : public ::testing::Test {};

TEST_F(FFTInpaintTest, DefaultConstructor) {
  FFTInpaint config;
  EXPECT_EQ(config.first_guess(), FirstGuess::kZonalAverage);
  EXPECT_TRUE(config.is_periodic());
  EXPECT_EQ(config.max_iterations(), 500);  // FFTInpaint sets this to 500
  EXPECT_DOUBLE_EQ(config.epsilon(), 1e-4);
  EXPECT_DOUBLE_EQ(config.sigma(), 10.0);
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(FFTInpaintTest, WithSigma) {
  FFTInpaint config;
  auto updated = config.with_sigma(20.0);
  EXPECT_DOUBLE_EQ(updated.sigma(), 20.0);
  EXPECT_DOUBLE_EQ(config.sigma(), 10.0);
}

TEST_F(FFTInpaintTest, WithFirstGuess) {
  FFTInpaint config;
  auto updated = config.with_first_guess(FirstGuess::kZero);
  EXPECT_EQ(updated.first_guess(), FirstGuess::kZero);
}

TEST_F(FFTInpaintTest, WithMaxIterations) {
  FFTInpaint config;
  auto updated = config.with_max_iterations(1000);
  EXPECT_EQ(updated.max_iterations(), 1000);
}

TEST_F(FFTInpaintTest, WithEpsilon) {
  FFTInpaint config;
  auto updated = config.with_epsilon(1e-5);
  EXPECT_DOUBLE_EQ(updated.epsilon(), 1e-5);
}

TEST_F(FFTInpaintTest, WithIsPeriodic) {
  FFTInpaint config;
  auto updated = config.with_is_periodic(false);
  EXPECT_FALSE(updated.is_periodic());
}

TEST_F(FFTInpaintTest, WithNumThreads) {
  FFTInpaint config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.num_threads(), 4);
}

TEST_F(FFTInpaintTest, MethodChaining) {
  FFTInpaint config;
  auto updated = config.with_sigma(15.0)
                     .with_first_guess(FirstGuess::kZero)
                     .with_max_iterations(800)
                     .with_epsilon(1e-5)
                     .with_num_threads(4);

  EXPECT_DOUBLE_EQ(updated.sigma(), 15.0);
  EXPECT_EQ(updated.first_guess(), FirstGuess::kZero);
  EXPECT_EQ(updated.max_iterations(), 800);
  EXPECT_DOUBLE_EQ(updated.epsilon(), 1e-5);
  EXPECT_EQ(updated.num_threads(), 4);
}

// ============================================================================
// GaussSeidel Config Tests
// ============================================================================

class GaussSeidelTest : public ::testing::Test {};

TEST_F(GaussSeidelTest, DefaultConstructor) {
  GaussSeidel config;
  EXPECT_EQ(config.first_guess(), FirstGuess::kZonalAverage);
  EXPECT_TRUE(config.is_periodic());
  EXPECT_EQ(config.max_iterations(), 2000);  // GaussSeidel sets this to 2000
  EXPECT_DOUBLE_EQ(config.epsilon(), 1e-4);
  EXPECT_DOUBLE_EQ(config.relaxation(), 1.0);
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(GaussSeidelTest, WithRelaxation) {
  GaussSeidel config;
  auto updated = config.with_relaxation(1.5);
  EXPECT_DOUBLE_EQ(updated.relaxation(), 1.5);
  EXPECT_DOUBLE_EQ(config.relaxation(), 1.0);
}

TEST_F(GaussSeidelTest, WithFirstGuess) {
  GaussSeidel config;
  auto updated = config.with_first_guess(FirstGuess::kZero);
  EXPECT_EQ(updated.first_guess(), FirstGuess::kZero);
}

TEST_F(GaussSeidelTest, WithMaxIterations) {
  GaussSeidel config;
  auto updated = config.with_max_iterations(3000);
  EXPECT_EQ(updated.max_iterations(), 3000);
}

TEST_F(GaussSeidelTest, WithEpsilon) {
  GaussSeidel config;
  auto updated = config.with_epsilon(1e-5);
  EXPECT_DOUBLE_EQ(updated.epsilon(), 1e-5);
}

TEST_F(GaussSeidelTest, WithIsPeriodic) {
  GaussSeidel config;
  auto updated = config.with_is_periodic(false);
  EXPECT_FALSE(updated.is_periodic());
}

TEST_F(GaussSeidelTest, WithNumThreads) {
  GaussSeidel config;
  auto updated = config.with_num_threads(8);
  EXPECT_EQ(updated.num_threads(), 8);
}

TEST_F(GaussSeidelTest, MethodChaining) {
  GaussSeidel config;
  auto updated = config.with_relaxation(1.8)
                     .with_first_guess(FirstGuess::kZero)
                     .with_max_iterations(5000)
                     .with_epsilon(1e-6)
                     .with_num_threads(16);

  EXPECT_DOUBLE_EQ(updated.relaxation(), 1.8);
  EXPECT_EQ(updated.first_guess(), FirstGuess::kZero);
  EXPECT_EQ(updated.max_iterations(), 5000);
  EXPECT_DOUBLE_EQ(updated.epsilon(), 1e-6);
  EXPECT_EQ(updated.num_threads(), 16);
}

// ============================================================================
// Multigrid Config Tests
// ============================================================================

class MultigridTest : public ::testing::Test {};

TEST_F(MultigridTest, DefaultConstructor) {
  Multigrid config;
  EXPECT_EQ(config.first_guess(), FirstGuess::kZonalAverage);
  EXPECT_TRUE(config.is_periodic());
  EXPECT_EQ(config.max_iterations(), 100);  // Multigrid sets this to 100
  EXPECT_DOUBLE_EQ(config.epsilon(), 1e-4);
  EXPECT_EQ(config.pre_smooth(), 2);
  EXPECT_EQ(config.post_smooth(), 2);
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(MultigridTest, WithPreSmooth) {
  Multigrid config;
  auto updated = config.with_pre_smooth(3);
  EXPECT_EQ(updated.pre_smooth(), 3);
  EXPECT_EQ(config.pre_smooth(), 2);
}

TEST_F(MultigridTest, WithPostSmooth) {
  Multigrid config;
  auto updated = config.with_post_smooth(4);
  EXPECT_EQ(updated.post_smooth(), 4);
  EXPECT_EQ(config.post_smooth(), 2);
}

TEST_F(MultigridTest, WithFirstGuess) {
  Multigrid config;
  auto updated = config.with_first_guess(FirstGuess::kZero);
  EXPECT_EQ(updated.first_guess(), FirstGuess::kZero);
}

TEST_F(MultigridTest, WithMaxIterations) {
  Multigrid config;
  auto updated = config.with_max_iterations(200);
  EXPECT_EQ(updated.max_iterations(), 200);
}

TEST_F(MultigridTest, WithEpsilon) {
  Multigrid config;
  auto updated = config.with_epsilon(1e-5);
  EXPECT_DOUBLE_EQ(updated.epsilon(), 1e-5);
}

TEST_F(MultigridTest, WithIsPeriodic) {
  Multigrid config;
  auto updated = config.with_is_periodic(false);
  EXPECT_FALSE(updated.is_periodic());
}

TEST_F(MultigridTest, WithNumThreads) {
  Multigrid config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.num_threads(), 4);
}

TEST_F(MultigridTest, MethodChaining) {
  Multigrid config;
  auto updated = config.with_pre_smooth(5)
                     .with_post_smooth(6)
                     .with_first_guess(FirstGuess::kZero)
                     .with_max_iterations(150)
                     .with_epsilon(1e-6)
                     .with_num_threads(8);

  EXPECT_EQ(updated.pre_smooth(), 5);
  EXPECT_EQ(updated.post_smooth(), 6);
  EXPECT_EQ(updated.first_guess(), FirstGuess::kZero);
  EXPECT_EQ(updated.max_iterations(), 150);
  EXPECT_DOUBLE_EQ(updated.epsilon(), 1e-6);
  EXPECT_EQ(updated.num_threads(), 8);
}

}  // namespace pyinterp::config::fill
