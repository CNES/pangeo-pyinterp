// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/math.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace pyinterp::math {

// Test fixture for math tests
class MathTest : public ::testing::Test {
 protected:
  static constexpr double kEpsilon = 1e-10;
};

// Tests for radians() function
TEST_F(MathTest, RadiansConversion) {
  EXPECT_DOUBLE_EQ(radians(0.0), 0.0);
  EXPECT_DOUBLE_EQ(radians(180.0), std::numbers::pi);
  EXPECT_DOUBLE_EQ(radians(90.0), std::numbers::pi / 2.0);
  EXPECT_DOUBLE_EQ(radians(360.0), 2.0 * std::numbers::pi);
  EXPECT_DOUBLE_EQ(radians(-90.0), -std::numbers::pi / 2.0);

  // Test with float
  EXPECT_FLOAT_EQ(radians(180.0f), std::numbers::pi_v<float>);
}

// Tests for degrees() function
TEST_F(MathTest, DegreesConversion) {
  EXPECT_DOUBLE_EQ(degrees(0.0), 0.0);
  EXPECT_DOUBLE_EQ(degrees(std::numbers::pi), 180.0);
  EXPECT_DOUBLE_EQ(degrees(std::numbers::pi / 2.0), 90.0);
  EXPECT_DOUBLE_EQ(degrees(2.0 * std::numbers::pi), 360.0);
  EXPECT_DOUBLE_EQ(degrees(-std::numbers::pi / 2.0), -90.0);

  // Test with float
  EXPECT_FLOAT_EQ(degrees(std::numbers::pi_v<float>), 180.0f);
}

// Tests for radians/degrees round trip
TEST_F(MathTest, RadiansDegreesRoundTrip) {
  const std::array<double, 5> test_values = {0.0, 45.0, 90.0, 180.0, 270.0};

  for (auto deg : test_values) {
    EXPECT_NEAR(degrees(radians(deg)), deg, kEpsilon)
        << "Round trip failed for " << deg << " degrees";
  }
}

// Tests for sqr() function
TEST_F(MathTest, SquareFunction) {
  EXPECT_EQ(sqr(0), 0);
  EXPECT_EQ(sqr(1), 1);
  EXPECT_EQ(sqr(2), 4);
  EXPECT_EQ(sqr(-3), 9);
  EXPECT_EQ(sqr(10), 100);

  EXPECT_DOUBLE_EQ(sqr(2.5), 6.25);
  EXPECT_DOUBLE_EQ(sqr(-1.5), 2.25);
}

// Tests for cosd() function
TEST_F(MathTest, CosineDegrees) {
  EXPECT_NEAR(cosd(0.0), 1.0, kEpsilon);
  EXPECT_NEAR(cosd(90.0), 0.0, kEpsilon);
  EXPECT_NEAR(cosd(180.0), -1.0, kEpsilon);
  EXPECT_NEAR(cosd(270.0), 0.0, kEpsilon);
  EXPECT_NEAR(cosd(360.0), 1.0, kEpsilon);
  EXPECT_NEAR(cosd(60.0), 0.5, kEpsilon);
}

// Tests for sind() function
TEST_F(MathTest, SineDegrees) {
  EXPECT_NEAR(sind(0.0), 0.0, kEpsilon);
  EXPECT_NEAR(sind(90.0), 1.0, kEpsilon);
  EXPECT_NEAR(sind(180.0), 0.0, kEpsilon);
  EXPECT_NEAR(sind(270.0), -1.0, kEpsilon);
  EXPECT_NEAR(sind(360.0), 0.0, kEpsilon);
  EXPECT_NEAR(sind(30.0), 0.5, kEpsilon);
}

// Tests for sincosd() function
TEST_F(MathTest, SineCosineDegrees) {
  auto [sin0, cos0] = sincosd(0.0);
  EXPECT_NEAR(sin0, 0.0, kEpsilon);
  EXPECT_NEAR(cos0, 1.0, kEpsilon);

  auto [sin90, cos90] = sincosd(90.0);
  EXPECT_NEAR(sin90, 1.0, kEpsilon);
  EXPECT_NEAR(cos90, 0.0, kEpsilon);

  auto [sin180, cos180] = sincosd(180.0);
  EXPECT_NEAR(sin180, 0.0, kEpsilon);
  EXPECT_NEAR(cos180, -1.0, kEpsilon);

  auto [sin45, cos45] = sincosd(45.0);
  EXPECT_NEAR(sin45, std::sqrt(2.0) / 2.0, kEpsilon);
  EXPECT_NEAR(cos45, std::sqrt(2.0) / 2.0, kEpsilon);
}

// Tests for sinc() function
TEST_F(MathTest, SincFunction) {
  // sinc(0) = 1
  EXPECT_DOUBLE_EQ(sinc(0.0), 1.0);

  // sinc(1) = sin(pi) / pi = 0
  EXPECT_NEAR(sinc(1.0), 0.0, kEpsilon);

  // sinc(0.5) = sin(pi/2) / (pi/2) = 2/pi
  EXPECT_NEAR(sinc(0.5), 2.0 / std::numbers::pi, kEpsilon);

  // Test negative values
  EXPECT_NEAR(sinc(-1.0), 0.0, kEpsilon);

  // Test with float
  EXPECT_FLOAT_EQ(sinc(0.0f), 1.0f);
}

// Tests for remainder() function with integral types
TEST_F(MathTest, RemainderIntegral) {
  EXPECT_EQ(remainder(7, 3), 1);
  EXPECT_EQ(remainder(8, 3), 2);
  EXPECT_EQ(remainder(9, 3), 0);

  // Negative dividend
  EXPECT_EQ(remainder(-7, 3), 2);  // -7 % 3 = -1, then -1 + 3 = 2
  EXPECT_EQ(remainder(-8, 3), 1);  // -8 % 3 = -2, then -2 + 3 = 1

  // Negative divisor
  EXPECT_EQ(remainder(7, -3), -2);   // 7 % -3 = 1, then 1 + (-3) = -2
  EXPECT_EQ(remainder(-7, -3), -1);  // -7 % -3 = -1 (already same sign)
}

// Tests for remainder() function with floating point types
TEST_F(MathTest, RemainderFloatingPoint) {
  EXPECT_NEAR(remainder(7.5, 3.0), 1.5, kEpsilon);
  EXPECT_NEAR(remainder(9.0, 3.0), 0.0, kEpsilon);

  // Negative values
  EXPECT_NEAR(remainder(-7.5, 3.0), 1.5, kEpsilon);
  EXPECT_NEAR(remainder(7.5, -3.0), -1.5, kEpsilon);
}

// Tests for normalize_period() function
TEST_F(MathTest, NormalizePeriod) {
  // Normalize to [0, 360)
  EXPECT_NEAR(normalize_period(0.0, 0.0, 360.0), 0.0, kEpsilon);
  EXPECT_NEAR(normalize_period(180.0, 0.0, 360.0), 180.0, kEpsilon);
  EXPECT_NEAR(normalize_period(360.0, 0.0, 360.0), 0.0, kEpsilon);
  EXPECT_NEAR(normalize_period(450.0, 0.0, 360.0), 90.0, kEpsilon);
  EXPECT_NEAR(normalize_period(-90.0, 0.0, 360.0), 270.0, kEpsilon);

  // Normalize to [-180, 180)
  EXPECT_NEAR(normalize_period(0.0, -180.0, 360.0), 0.0, kEpsilon);
  EXPECT_NEAR(normalize_period(180.0, -180.0, 360.0), -180.0, kEpsilon);
  EXPECT_NEAR(normalize_period(-180.0, -180.0, 360.0), -180.0, kEpsilon);

  // Test with integral types
  EXPECT_EQ(normalize_period(10, 0, 8), 2);
}

// Tests for normalize_period_half() function
TEST_F(MathTest, NormalizePeriodHalf) {
  // Normalize to [-180, 180)
  EXPECT_NEAR(normalize_period_half(0.0, 0.0, 360.0), 0.0, kEpsilon);
  EXPECT_NEAR(normalize_period_half(90.0, 0.0, 360.0), 90.0, kEpsilon);
  EXPECT_NEAR(normalize_period_half(180.0, 0.0, 360.0), -180.0, kEpsilon);
  EXPECT_NEAR(normalize_period_half(270.0, 0.0, 360.0), -90.0, kEpsilon);
  EXPECT_NEAR(normalize_period_half(-90.0, 0.0, 360.0), -90.0, kEpsilon);

  // Test values already in range
  EXPECT_NEAR(normalize_period_half(45.0, 0.0, 360.0), 45.0, kEpsilon);
  EXPECT_NEAR(normalize_period_half(-45.0, 0.0, 360.0), -45.0, kEpsilon);

  // Test with integral types
  EXPECT_EQ(normalize_period_half(10, 0, 8), 2);
  EXPECT_EQ(normalize_period_half(3, 0, 8), 3);
}

// Tests for is_same() function with integral types
TEST_F(MathTest, IsSameIntegral) {
  EXPECT_TRUE(is_same(5, 5, 0));
  EXPECT_TRUE(is_same(5, 6, 1));
  EXPECT_TRUE(is_same(5, 4, 1));
  EXPECT_FALSE(is_same(5, 7, 1));

  EXPECT_TRUE(is_same(-5, -5, 0));
  EXPECT_TRUE(is_same(-5, -6, 1));
}

// Tests for is_same() function with floating point types
TEST_F(MathTest, IsSameFloatingPoint) {
  EXPECT_TRUE(is_same(1.0, 1.0, kEpsilon));
  EXPECT_TRUE(is_same(1.0, 1.0 + kEpsilon / 2.0, kEpsilon));
  EXPECT_FALSE(is_same(1.0, 1.1, kEpsilon));

  // Test with very small numbers
  EXPECT_TRUE(is_same(1e-15, 1e-15, kEpsilon));

  // Test with very large numbers
  EXPECT_TRUE(is_same(1e15, 1e15 + 1.0, 1e5));
}

// Tests for is_almost_zero() function
TEST_F(MathTest, IsAlmostZero) {
  EXPECT_TRUE(is_almost_zero(0.0));
  EXPECT_TRUE(is_almost_zero(1e-20));
  EXPECT_TRUE(is_almost_zero(-1e-20));
  EXPECT_FALSE(is_almost_zero(0.1));
  EXPECT_FALSE(is_almost_zero(-0.1));

  // Test with custom epsilon
  EXPECT_TRUE(is_almost_zero(0.01, 0.1));
  EXPECT_FALSE(is_almost_zero(0.2, 0.1));

  // Test with float
  EXPECT_TRUE(is_almost_zero(0.0f));
}

// Tests for pow<N>() function
TEST_F(MathTest, CompileTimePower) {
  // Test with N = 0
  EXPECT_EQ(pow<0>(5), 1);
  EXPECT_EQ(pow<0>(10.0), 1.0);

  // Test with N = 1
  EXPECT_EQ(pow<1>(5), 5);
  EXPECT_EQ(pow<1>(10.0), 10.0);

  // Test with N = 2
  EXPECT_EQ(pow<2>(5), 25);
  EXPECT_EQ(pow<2>(3.0), 9.0);

  // Test with N = 3
  EXPECT_EQ(pow<3>(2), 8);
  EXPECT_EQ(pow<3>(3.0), 27.0);

  // Test with N = 4
  EXPECT_EQ(pow<4>(2), 16);
  EXPECT_EQ(pow<4>(3.0), 81.0);

  // Test with N = 5
  EXPECT_EQ(pow<5>(2), 32);

  // Test with N = 10
  EXPECT_EQ(pow<10>(2), 1024);

  // Test with floating point
  EXPECT_DOUBLE_EQ(pow<2>(2.5), 6.25);
  EXPECT_DOUBLE_EQ(pow<3>(2.0), 8.0);
}

// Tests for power2() function
TEST_F(MathTest, Power2Function) {
  EXPECT_DOUBLE_EQ(power2(0), 1.0);
  EXPECT_DOUBLE_EQ(power2(1), 2.0);
  EXPECT_DOUBLE_EQ(power2(2), 4.0);
  EXPECT_DOUBLE_EQ(power2(3), 8.0);
  EXPECT_DOUBLE_EQ(power2(10), 1024.0);

  // Test negative exponents
  EXPECT_DOUBLE_EQ(power2(-1), 0.5);
  EXPECT_DOUBLE_EQ(power2(-2), 0.25);
  EXPECT_DOUBLE_EQ(power2(-10), 1.0 / 1024.0);
}

// Tests for power10() function
TEST_F(MathTest, Power10Function) {
  EXPECT_DOUBLE_EQ(power10(0), 1.0);
  EXPECT_DOUBLE_EQ(power10(1), 10.0);
  EXPECT_DOUBLE_EQ(power10(2), 100.0);
  EXPECT_DOUBLE_EQ(power10(3), 1000.0);
  EXPECT_DOUBLE_EQ(power10(6), 1e6);

  // Test negative exponents
  EXPECT_DOUBLE_EQ(power10(-1), 0.1);
  EXPECT_DOUBLE_EQ(power10(-2), 0.01);
  EXPECT_DOUBLE_EQ(power10(-3), 0.001);

  // Compare with std::pow for accuracy
  for (int exp = -10; exp <= 10; ++exp) {
    EXPECT_NEAR(power10(exp), std::pow(10.0, exp), 1e-10)
        << "Mismatch for exponent " << exp;
  }
}

// Constexpr tests
TEST(MathConstexprTest, RadiansConstexpr) {
  constexpr double rad = radians(180.0);
  EXPECT_DOUBLE_EQ(rad, std::numbers::pi);
}

TEST(MathConstexprTest, DegreesConstexpr) {
  constexpr double deg = degrees(std::numbers::pi);
  EXPECT_DOUBLE_EQ(deg, 180.0);
}

TEST(MathConstexprTest, SqrConstexpr) {
  constexpr int square = sqr(5);
  EXPECT_EQ(square, 25);
}

TEST(MathConstexprTest, RemainderConstexpr) {
  constexpr int rem = remainder(7, 3);
  EXPECT_EQ(rem, 1);
}

TEST(MathConstexprTest, NormalizePeriodConstexpr) {
  auto norm = normalize_period(450.0, 0.0, 360.0);
  EXPECT_NEAR(norm, 90.0, 1e-10);
}

TEST(MathConstexprTest, NormalizePeriodHalfConstexpr) {
  auto norm = normalize_period_half(270.0, 0.0, 360.0);
  EXPECT_NEAR(norm, -90.0, 1e-10);
}

TEST(MathConstexprTest, IsSameConstexpr) {
  auto same = is_same(5, 6, 1);
  EXPECT_TRUE(same);
}

TEST(MathConstexprTest, IsAlmostZeroConstexpr) {
  auto zero = is_almost_zero(1e-20);
  EXPECT_TRUE(zero);
}

TEST(MathConstexprTest, PowConstexpr) {
  constexpr int result = pow<5>(2);
  EXPECT_EQ(result, 32);
}

TEST(MathConstexprTest, Power2Constexpr) {
  auto result = power2(5);
  EXPECT_DOUBLE_EQ(result, 32.0);
}

TEST(MathConstexprTest, Power10Constexpr) {
  auto result = power10(3);
  EXPECT_DOUBLE_EQ(result, 1000.0);
}

// Edge case tests
TEST(MathEdgeCaseTest, SincEdgeCases) {
  // Test very small values
  EXPECT_NEAR(sinc(1e-10), 1.0, 1e-9);

  // Test values where sin(pi*x) crosses zero
  EXPECT_NEAR(sinc(1.0), 0.0, 1e-10);
  EXPECT_NEAR(sinc(2.0), 0.0, 1e-10);
}

TEST(MathEdgeCaseTest, NormalizePeriodEdgeCases) {
  // Test with very large values
  EXPECT_NEAR(normalize_period(1e6, 0.0, 360.0), std::fmod(1e6, 360.0), 1e-6);

  // Test boundary values
  EXPECT_NEAR(normalize_period(359.999, 0.0, 360.0), 359.999, 1e-10);
}

TEST(MathEdgeCaseTest, IsSameEdgeCases) {
  // Test with NaN (should return false)
  EXPECT_FALSE(is_same(std::numeric_limits<double>::quiet_NaN(), 0.0, 1e-10));

  // Test with infinity
  EXPECT_FALSE(is_same(std::numeric_limits<double>::infinity(), 1e100, 1e10));
}

}  // namespace pyinterp::math
