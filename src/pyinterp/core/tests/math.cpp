// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/detail/math.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

namespace math = pyinterp::detail::math;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

TEST(math, pi) {
  // pi<double> == π
  EXPECT_DOUBLE_EQ(math::pi<double>(), M_PI);
}

TEST(math, pi_2) {
  // pi_2<double> == π / 2
  EXPECT_DOUBLE_EQ(math::pi_2<double>(), M_PI * 0.5);
}

TEST(math, two_pi) {
  // two_pi<double> == 2π
  EXPECT_DOUBLE_EQ(math::two_pi<double>(), 2 * M_PI);
}

TEST(math, sqr) {
  // sqr(a) == (a * a)
  EXPECT_DOUBLE_EQ(math::sqr(math::pi<double>()), M_PI * M_PI);
}

TEST(math, normalize_angle) {
  // normalize_angle(x + kπ) == x
  EXPECT_NEAR(math::normalize_angle(720.001, -180.0, 360.0), 0.001, 1e-12);
  EXPECT_DOUBLE_EQ(math::normalize_angle(180.0, -180.0, 360.0), -180.0);
  EXPECT_DOUBLE_EQ(math::normalize_angle(2.5 * M_PI, -M_PI, 2 * M_PI),
                   M_PI * 0.5);
}

TEST(math, remainder) {
  // x % y like Python
  EXPECT_EQ(math::remainder(360, 181), 179);
  EXPECT_EQ(math::remainder(360, -181), -2);
}

TEST(math, sind) {
  // sind(x) == sin(x * π / 180)
  for (double x = -720; x <= 720; x += 0.1) {
    EXPECT_NEAR(math::sind(x), std::sin(math::radians(x)), 1e-12);
  }
}

TEST(math, cosd) {
  // cosd(x) == cos(x * π / 180)
  for (double x = -720; x <= 720; x += 0.1) {
    EXPECT_NEAR(math::cosd(x), std::cos(math::radians(x)), 1e-12);
  }
}

TEST(math, sincosd) {
  // sincosd(x) == sin(x * π / 180), cos(x * π / 180)
  for (double x = -720; x <= 720; x += 0.1) {
    auto [sinx, cosx] = math::sincosd(x);
    EXPECT_NEAR(sinx, std::sin(math::radians(x)), 1e-12);
    EXPECT_NEAR(cosx, std::cos(math::radians(x)), 1e-12);
  }
}

TEST(math, tand) {
  // tand(x) == tan(x * π / 180)
  for (double x = -720; x <= 720; x += 0.1) {
    if (std::remainder(x, 90) > 1e-9) {
      EXPECT_NEAR(math::tand(x), std::tan(math::radians(x)), 1e-9);
    }
  }
}

TEST(math, atan2d) {
  // atan2d(x, y) == atan2(x, y) * 180 / π
  std::uniform_real_distribution<double> dist(-1000, 1000);
  std::default_random_engine re;
  for (int64_t ix = 0; ix < 2000; ++ix) {
    double x = dist(re), y = dist(re);
    EXPECT_NEAR(math::atan2d(x, y), math::degrees(std::atan2(x, y)), 1e-9);
  }
}

TEST(math, atand) {
  // math::atand(x) == atan(x) * 180 / π
  std::uniform_real_distribution<double> dist(-1000, 1000);
  std::default_random_engine re;
  for (int64_t ix = 0; ix < 2000; ++ix) {
    double x = dist(re);
    EXPECT_NEAR(math::atand(x), math::degrees(std::atan(x)), 1e-9);
  }
}

TEST(math, is_same) {
  EXPECT_TRUE(math::is_same<double>(M_PI, math::pi<float>(), 1e-6));
  EXPECT_FALSE(math::is_same<double>(M_PI, math::pi<float>(), 1e-12));

  EXPECT_TRUE(math::is_same<int64_t>(1, 2, 1));
  EXPECT_FALSE(math::is_same<int64_t>(1, 2, 0));
}

TEST(math, fill_value) {
  auto float_ = math::Fill<float>();
  EXPECT_TRUE(std::isnan(float_.value()));
  EXPECT_TRUE(float_.is(float_.value()));
  EXPECT_TRUE(float_.is(std::numeric_limits<float>::quiet_NaN()));
  EXPECT_FALSE(float_.is(1));
  EXPECT_FALSE(float_.is_not(float_.value()));
  EXPECT_FALSE(float_.is_not(std::numeric_limits<float>::quiet_NaN()));
  EXPECT_TRUE(float_.is_not(1));

  auto int_ = math::Fill<int>();
  EXPECT_EQ(int_.value(), std::numeric_limits<int>::max());
  EXPECT_TRUE(int_.is(int_.value()));
  EXPECT_TRUE(int_.is(std::numeric_limits<int>::max()));
  EXPECT_FALSE(int_.is(1));
  EXPECT_FALSE(int_.is_not(int_.value()));
  EXPECT_FALSE(int_.is_not(std::numeric_limits<int>::max()));
  EXPECT_TRUE(int_.is_not(1));
}

TEST(math, is_within) {
  double step = 0.1;
  double value = 26.0 + 58 * step;
  EXPECT_TRUE(math::is_within(value, 31.8, 1));

  float from = 0;
  float to =
      std::nextafter(std::nextafter(std::nextafter(from, 1.f), 1.f), 1.f);
  EXPECT_FALSE(math::is_within(from, to, 1));
  EXPECT_FALSE(math::is_within(from, to, 2));
  EXPECT_TRUE(math::is_within(from, to, 3));
}

TEST(math, is_almost_zero) {
  EXPECT_TRUE(
      math::is_almost_zero(0.0, std::numeric_limits<double>::epsilon()));
  EXPECT_FALSE(
      math::is_almost_zero(0.0 + std::numeric_limits<double>::epsilon(),
                           std::numeric_limits<double>::epsilon()));
}

TEST(math, sinc) {
  EXPECT_FLOAT_EQ(math::sinc(0.0), 1.0);
  EXPECT_NEAR(math::sinc(0.1), 0.983631643083466, 1e-6);
  EXPECT_NEAR(math::sinc(0.5), 0.6366197723675814, 1e-6);
}

TEST(math, power2) {
  EXPECT_DOUBLE_EQ(math::power2(0.0), 1.0);
  EXPECT_DOUBLE_EQ(math::power2(1.0), 2.0);
  EXPECT_DOUBLE_EQ(math::power2(16.0), 65536.0);
  EXPECT_DOUBLE_EQ(math::power2(32.0), 4294967296.0);
  EXPECT_DOUBLE_EQ(math::power2(-16.0), 1.52587890625e-05);
}

TEST(math, power10) {
  EXPECT_DOUBLE_EQ(math::power10(0), 1.0);
  EXPECT_DOUBLE_EQ(math::power10(1), 10.0);
  EXPECT_DOUBLE_EQ(math::power10(10), 1e10);
  EXPECT_DOUBLE_EQ(math::power10(11), 1e11);
  EXPECT_DOUBLE_EQ(math::power10(-11), 1e-11);
  EXPECT_DOUBLE_EQ(math::power10(-10), 1e-10);
}
