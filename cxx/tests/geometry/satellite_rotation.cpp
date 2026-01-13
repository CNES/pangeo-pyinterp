// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <numbers>

#include "pyinterp/geometry/satellite/rotation.hpp"

namespace satellite = pyinterp::geometry::satellite;

// Test fixture for satellite rotation tests
class SatelliteRotationTest : public ::testing::Test {
 protected:
  static constexpr double kEpsilon = 1e-10;
};

// Test satellite_direction with simple linear movement
TEST_F(SatelliteRotationTest, SatelliteDirectionLinear) {
  // Create a simple linear path along X-axis
  Eigen::Matrix<double, Eigen::Dynamic, 3> location(5, 3);
  location << 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0,
      0.0, 0.0;

  auto direction = satellite::rotation::satellite_direction<double>(location);

  // Should have same dimensions as input
  EXPECT_EQ(direction.rows(), 5);
  EXPECT_EQ(direction.cols(), 3);

  // All directions should point along X-axis (1, 0, 0)
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(direction(i, 0), 1.0, kEpsilon)
        << "X component wrong at row " << i;
    EXPECT_NEAR(direction(i, 1), 0.0, kEpsilon)
        << "Y component wrong at row " << i;
    EXPECT_NEAR(direction(i, 2), 0.0, kEpsilon)
        << "Z component wrong at row " << i;

    // Should be unit vector
    double norm = direction.row(i).norm();
    EXPECT_NEAR(norm, 1.0, kEpsilon) << "Not unit vector at row " << i;
  }
}

// Test satellite_direction with diagonal movement
TEST_F(SatelliteRotationTest, SatelliteDirectionDiagonal) {
  // Create movement along diagonal (1, 1, 1) direction
  Eigen::Matrix<double, Eigen::Dynamic, 3> location(4, 3);
  location << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0;

  auto direction = satellite::rotation::satellite_direction<double>(location);

  // Expected direction is (1, 1, 1) normalized = (1/√3, 1/√3, 1/√3)
  double expected_component = std::numbers::inv_sqrt3;

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(direction(i, 0), expected_component, kEpsilon)
        << "X component wrong at row " << i;
    EXPECT_NEAR(direction(i, 1), expected_component, kEpsilon)
        << "Y component wrong at row " << i;
    EXPECT_NEAR(direction(i, 2), expected_component, kEpsilon)
        << "Z component wrong at row " << i;

    // Should be unit vector
    double norm = direction.row(i).norm();
    EXPECT_NEAR(norm, 1.0, kEpsilon) << "Not unit vector at row " << i;
  }
}

// Test satellite_direction boundary handling
TEST_F(SatelliteRotationTest, SatelliteDirectionBoundaries) {
  // Create a path with varying direction
  Eigen::Matrix<double, Eigen::Dynamic, 3> location(5, 3);
  location << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0, 2.0, 0.0, 4.0,
      3.0, 0.0;

  auto direction = satellite::rotation::satellite_direction<double>(location);

  // First and last rows should be copied from their neighbors
  EXPECT_TRUE(direction.row(0).isApprox(direction.row(1), kEpsilon))
      << "First row not copied from second row";
  EXPECT_TRUE(direction.row(4).isApprox(direction.row(3), kEpsilon))
      << "Last row not copied from second-to-last row";

  // All should be unit vectors
  for (int i = 0; i < 5; ++i) {
    double norm = direction.row(i).norm();
    EXPECT_NEAR(norm, 1.0, kEpsilon) << "Not unit vector at row " << i;
  }
}

// Test satellite_direction with minimum points (3)
TEST_F(SatelliteRotationTest, SatelliteDirectionMinimumPoints) {
  Eigen::Matrix<double, Eigen::Dynamic, 3> location(3, 3);
  location << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0;

  auto direction = satellite::rotation::satellite_direction<double>(location);

  EXPECT_EQ(direction.rows(), 3);

  // All should point along X-axis
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(direction(i, 0), 1.0, kEpsilon);
    EXPECT_NEAR(direction(i, 1), 0.0, kEpsilon);
    EXPECT_NEAR(direction(i, 2), 0.0, kEpsilon);
  }
}

// Test matrix_3d with 90-degree rotation around Z-axis
TEST_F(SatelliteRotationTest, Matrix3DRotationZAxis90Degrees) {
  Eigen::Matrix<double, 3, 1> z_axis(0.0, 0.0, 1.0);
  double theta = std::numbers::pi / 2.0;  // 90 degrees

  auto rotation = satellite::rotation::matrix_3d<double>(theta, z_axis);

  // Should be 3x3
  EXPECT_EQ(rotation.rows(), 3);
  EXPECT_EQ(rotation.cols(), 3);

  // Apply rotation to X-axis vector (1, 0, 0)
  Eigen::Matrix<double, 3, 1> x_vec(1.0, 0.0, 0.0);
  Eigen::Matrix<double, 3, 1> rotated = rotation * x_vec;

  // Should become (0, -1, 0) because axis is negated in the function
  EXPECT_NEAR(rotated(0), 0.0, kEpsilon);
  EXPECT_NEAR(rotated(1), -1.0, kEpsilon);
  EXPECT_NEAR(rotated(2), 0.0, kEpsilon);
}

// Test matrix_3d with 180-degree rotation
TEST_F(SatelliteRotationTest, Matrix3DRotation180Degrees) {
  Eigen::Matrix<double, 3, 1> z_axis(0.0, 0.0, 1.0);
  double theta = std::numbers::pi;  // 180 degrees

  auto rotation = satellite::rotation::matrix_3d<double>(theta, z_axis);

  // Apply rotation to X-axis vector
  Eigen::Matrix<double, 3, 1> x_vec(1.0, 0.0, 0.0);
  Eigen::Matrix<double, 3, 1> rotated = rotation * x_vec;

  // Should become (-1, 0, 0)
  EXPECT_NEAR(rotated(0), -1.0, kEpsilon);
  EXPECT_NEAR(rotated(1), 0.0, kEpsilon);
  EXPECT_NEAR(rotated(2), 0.0, kEpsilon);
}

// Test matrix_3d with arbitrary axis rotation
TEST_F(SatelliteRotationTest, Matrix3DArbitraryAxis) {
  // Rotate around (1, 1, 0) axis by 90 degrees
  Eigen::Matrix<double, 3, 1> axis(1.0, 1.0, 0.0);
  double theta = std::numbers::pi / 2.0;

  auto rotation = satellite::rotation::matrix_3d<double>(theta, axis);

  // Rotation matrix should be orthogonal (R^T * R = I)
  auto product = rotation.transpose() * rotation;
  EXPECT_TRUE(product.isApprox(Eigen::Matrix3d::Identity(), kEpsilon))
      << "Rotation matrix not orthogonal";

  // Determinant should be 1 (proper rotation, not reflection)
  EXPECT_NEAR(rotation.determinant(), 1.0, kEpsilon);
}

// Test matrix_3d preserves vector length
TEST_F(SatelliteRotationTest, Matrix3DPreservesLength) {
  Eigen::Matrix<double, 3, 1> axis(1.0, 2.0, 3.0);
  double theta = 0.7;  // arbitrary angle

  auto rotation = satellite::rotation::matrix_3d<double>(theta, axis);

  // Test with multiple vectors
  std::vector<Eigen::Matrix<double, 3, 1>> test_vectors = {{1.0, 0.0, 0.0},
                                                           {0.0, 1.0, 0.0},
                                                           {0.0, 0.0, 1.0},
                                                           {1.0, 1.0, 1.0},
                                                           {2.5, -1.3, 4.2}};

  for (const auto& vec : test_vectors) {
    auto rotated = rotation * vec;
    EXPECT_NEAR(vec.norm(), rotated.norm(), kEpsilon)
        << "Rotation changed vector length";
  }
}

// Test matrix_3d with zero rotation
TEST_F(SatelliteRotationTest, Matrix3DZeroRotation) {
  Eigen::Matrix<double, 3, 1> axis(1.0, 0.0, 0.0);
  double theta = 0.0;

  auto rotation = satellite::rotation::matrix_3d<double>(theta, axis);

  // Should be identity matrix
  EXPECT_TRUE(rotation.isApprox(Eigen::Matrix3d::Identity(), kEpsilon));
}

// Test matrix_3d with non-unit axis (should auto-normalize)
TEST_F(SatelliteRotationTest, Matrix3DNonUnitAxis) {
  // Provide a non-normalized axis
  Eigen::Matrix<double, 3, 1> axis(2.0, 0.0, 0.0);  // Length = 2
  double theta = std::numbers::pi / 2.0;

  auto rotation = satellite::rotation::matrix_3d<double>(theta, axis);

  // Should still be a valid rotation matrix
  EXPECT_NEAR(rotation.determinant(), 1.0, kEpsilon);
  auto product = rotation.transpose() * rotation;
  EXPECT_TRUE(product.isApprox(Eigen::Matrix3d::Identity(), kEpsilon));
}

// Test matrix_3d rotation composition
TEST_F(SatelliteRotationTest, Matrix3DComposition) {
  Eigen::Matrix<double, 3, 1> z_axis(0.0, 0.0, 1.0);

  // Two 90-degree rotations should equal one 180-degree rotation
  auto rot90 =
      satellite::rotation::matrix_3d<double>(std::numbers::pi / 2.0, z_axis);
  auto rot180 =
      satellite::rotation::matrix_3d<double>(std::numbers::pi, z_axis);

  auto composed = rot90 * rot90;

  EXPECT_TRUE(composed.isApprox(rot180, kEpsilon))
      << "Rotation composition failed";
}

// Test matrix_3d inverse rotation
TEST_F(SatelliteRotationTest, Matrix3DInverseRotation) {
  Eigen::Matrix<double, 3, 1> axis(1.0, 1.0, 1.0);
  double theta = 0.5;

  auto rotation = satellite::rotation::matrix_3d<double>(theta, axis);
  auto inverse = satellite::rotation::matrix_3d<double>(-theta, axis);

  // R * R^-1 should be identity
  auto product = rotation * inverse;
  EXPECT_TRUE(product.isApprox(Eigen::Matrix3d::Identity(), kEpsilon));
}

// Test with float precision
TEST_F(SatelliteRotationTest, FloatPrecision) {
  Eigen::Matrix<float, Eigen::Dynamic, 3> location(3, 3);
  location << 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f;

  auto direction = satellite::rotation::satellite_direction<float>(location);

  EXPECT_EQ(direction.rows(), 3);
  EXPECT_EQ(direction.cols(), 3);

  // Test rotation with float
  Eigen::Matrix<float, 3, 1> axis(0.0f, 0.0f, 1.0f);
  auto rotation = satellite::rotation::matrix_3d<float>(
      std::numbers::pi_v<float> / 2.0f, axis);

  EXPECT_EQ(rotation.rows(), 3);
  EXPECT_EQ(rotation.cols(), 3);
}
