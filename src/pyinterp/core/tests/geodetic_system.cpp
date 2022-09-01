// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include <random>

#include "pyinterp/detail/geodetic/spheroid.hpp"

namespace geodetic = pyinterp::detail::geodetic;

TEST(geometry_geodetic_system, wgs84) {
  // WGS-84.
  auto wgs84 = geodetic::Spheroid();
  // https://fr.wikipedia.org/wiki/WGS_84
  // https://en.wikipedia.org/wiki/Geodetic_datum
  // http://earth-info.nga.mil/GandG/publications/tr8350.2/wgs84fin.pdf
  EXPECT_DOUBLE_EQ(wgs84.semi_major_axis(), 6'378'137);
  EXPECT_DOUBLE_EQ(wgs84.flattening(), 1 / 298.257'223'563);
  EXPECT_DOUBLE_EQ(wgs84.semi_minor_axis(), 6'356'752.314'245'179'497'563'967);
  EXPECT_NEAR(std::sqrt(wgs84.first_eccentricity_squared()),
              0.081'819'190'842'622, 1e-15);
  EXPECT_NEAR(std::sqrt(wgs84.second_eccentricity_squared()),
              8.2'094'437'949'696 * 1e-2, 1e-15);
  EXPECT_NEAR(wgs84.equatorial_circumference(true) * 1e-3, 40'075.017, 1e-3);
  EXPECT_NEAR(wgs84.equatorial_circumference(false) * 1e-3, 39'940.652, 1e-3);
  EXPECT_NEAR(wgs84.polar_radius_of_curvature(), 6399593.6258, 1e-4);
  EXPECT_NEAR(wgs84.equatorial_radius_of_curvature(), 6335439.3272, 1e-4);
  EXPECT_NEAR(wgs84.axis_ratio(), 0.996647189335, 1e-12);
  EXPECT_NEAR(wgs84.linear_eccentricity(), 5.2185400842339 * 1E5, 1e-6);
  EXPECT_NEAR(wgs84.mean_radius(), 6371008.7714, 1e-4);
  EXPECT_NEAR(wgs84.authalic_radius(), 6371007.1809, 1e-4);
  EXPECT_NEAR(wgs84.volumetric_radius(), 6371000.7900, 1e-4);
  EXPECT_EQ(static_cast<std::string>(wgs84),
            "Spheroid(a=6378137, b=6356752.31, f=0.00335281066)");
}

TEST(geometry_geodetic_system, operator) {
  auto wgs84 = geodetic::Spheroid();
  // https://en.wikipedia.org/wiki/Geodetic_Reference_System_1980
  auto grs80 = geodetic::Spheroid(6'378'137, 1 / 298.257'222'101);
  EXPECT_DOUBLE_EQ(grs80.semi_major_axis(), 6'378'137);
  EXPECT_DOUBLE_EQ(grs80.flattening(), 1 / 298.257'222'101);
  EXPECT_EQ(wgs84, wgs84);
  EXPECT_NE(wgs84, grs80);
  EXPECT_EQ(static_cast<std::string>(grs80),
            "Spheroid(a=6378137, b=6356752.31, f=0.00335281068)");
}
