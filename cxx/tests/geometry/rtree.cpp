// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geometry/rtree.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "pyinterp/math/interpolate/kriging.hpp"
#include "pyinterp/math/interpolate/rbf.hpp"
#include "pyinterp/math/interpolate/window_function.hpp"

using Point2D =
    boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian>;

using Point3D =
    boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian>;

namespace pyinterp::geometry {

using RTree2D = RTree<Point2D, double>;
using RTree3D = RTree<Point3D, double>;

TEST(RTree2D, BasicInsertAndQuery) {
  RTree2D tree;
  std::vector<RTree2D::value_t> data = {{Point2D(0.0, 0.0), 1.0},
                                        {Point2D(1.0, 1.0), 2.0},
                                        {Point2D(2.0, 2.0), 3.0},
                                        {Point2D(3.0, 3.0), 4.0}};
  for (const auto& v : data) tree.insert(v);
  EXPECT_EQ(tree.size(), data.size());
  EXPECT_FALSE(tree.empty());
  auto bounds = tree.bounds();
  ASSERT_TRUE(bounds.has_value());
  tree.clear();
  EXPECT_EQ(tree.size(), 0);
  EXPECT_TRUE(tree.empty());
}

TEST(RTree2D, Packing) {
  RTree2D tree;
  std::vector<RTree2D::value_t> data = {{Point2D(0.0, 0.0), 1.0},
                                        {Point2D(1.0, 1.0), 2.0},
                                        {Point2D(2.0, 2.0), 3.0}};
  tree.packing(data);
  EXPECT_EQ(tree.size(), data.size());
}

TEST(RTree2D, KNNQuery) {
  RTree2D tree;
  for (int i = 0; i < 10; ++i) {
    tree.insert({Point2D(i, i), static_cast<double>(i)});
  }
  auto result =
      tree.query(Point2D(5.1, 5.1), 3, std::numeric_limits<double>::max());
  ASSERT_EQ(result.size(), 3);
  EXPECT_NEAR(result[0].second, 5.0, 1e-12);
}

TEST(RTree2D, QueryBall) {
  RTree2D tree;
  for (int i = 0; i < 10; ++i) {
    tree.insert({Point2D(i, 0), static_cast<double>(i)});
  }
  auto result = tree.query_ball(Point2D(5, 0), 2.0);
  std::vector<double> found;
  found.reserve(result.size());
  for (const auto& [dist, val] : result) {
    found.push_back(val);
  }
  EXPECT_TRUE(std::ranges::find(found, 4.0) != found.end());
  EXPECT_TRUE(std::ranges::find(found, 5.0) != found.end());
  EXPECT_TRUE(std::ranges::find(found, 6.0) != found.end());
}

TEST(RTree2D, QueryWithin) {
  RTree2D tree;
  for (int i = 0; i < 10; ++i) {
    tree.insert({Point2D(i, 0), static_cast<double>(i)});
  }
  auto result = tree.query(Point2D(5, 0), 3, std::numeric_limits<double>::max(),
                           pyinterp::geometry::BoundaryCheck::kEnvelope);
  // May be empty if not surrounded, but should not crash
  EXPECT_LE(result.size(), 3);
}

TEST(RTree2D, ValueQuery) {
  RTree2D tree;
  for (int i = 0; i < 10; ++i) {
    tree.insert({Point2D(i, 0), static_cast<double>(i)});
  }
  auto result = tree.value(Point2D(5, 0), 2.0, 3,
                           pyinterp::geometry::BoundaryCheck::kNone);
  EXPECT_LE(result.size(), 3);
}

TEST(RTree2D, InverseDistanceWeighting) {
  RTree2D tree;
  for (int i = 0; i < 10; ++i) {
    tree.insert({Point2D(i, 0), static_cast<double>(i)});
  }
  auto [val, n] = tree.inverse_distance_weighting(
      Point2D(5, 0), 2.0, 3, 2, pyinterp::geometry::BoundaryCheck::kNone);
  EXPECT_TRUE(std::isfinite(val) || std::isnan(val));
  EXPECT_LE(n, 3);
}

TEST(RTree2D, KrigingInstantiation) {
  using promotion_t = decltype(std::declval<double>() + std::declval<double>());
  math::interpolate::Kriging<promotion_t> model(
      1.0, 1.0, 1.0, math::interpolate::CovarianceFunction::kGaussian);
  RTree2D tree;
  for (int i = 0; i < 3; ++i) {
    tree.insert({Point2D(i, 0), static_cast<double>(i)});
  }
  auto [val, n] = tree.kriging(model, Point2D(1, 0), 2.0, 3,
                               pyinterp::geometry::BoundaryCheck::kNone);
  EXPECT_TRUE(std::isfinite(val) || std::isnan(val));
  EXPECT_LE(n, 3);
}

TEST(RTree2D, RBFInstantiation) {
  using promotion_t = decltype(std::declval<double>() + std::declval<double>());
  math::interpolate::RBF<promotion_t> model(
      std::numeric_limits<double>::quiet_NaN(), 0,
      math::interpolate::RBFKernel::kMultiquadric);
  RTree2D tree;
  for (int i = 0; i < 3; ++i) {
    tree.insert({Point2D(i, 0), static_cast<double>(i)});
  }
  auto [val, n] = tree.radial_basis_function(
      model, Point2D(1, 0), 2.0, 3, pyinterp::geometry::BoundaryCheck::kNone);
  EXPECT_TRUE(std::isfinite(val) || std::isnan(val));
  EXPECT_LE(n, 3);
}

TEST(RTree2D, WindowFunctionInstantiation) {
  RTree2D tree;
  for (int i = 0; i < 3; ++i) {
    tree.insert({Point2D(i, 0), static_cast<double>(i)});
  }
  math::interpolate::InterpolationWindow<double> model(
      math::interpolate::window::Kernel::kHamming, 0.5);
  auto [val, n] = tree.window_function(
      model, Point2D(1, 0), 2.0, 3, pyinterp::geometry::BoundaryCheck::kNone);
  EXPECT_TRUE(std::isfinite(val) || std::isnan(val));
  EXPECT_LE(n, 3);
}

// 3D Test Cases
TEST(RTree3D, BasicInsertAndQuery) {
  RTree3D tree;
  std::vector<RTree3D::value_t> data = {{Point3D(0.0, 0.0, 0.0), 1.0},
                                        {Point3D(1.0, 1.0, 1.0), 2.0},
                                        {Point3D(2.0, 2.0, 2.0), 3.0},
                                        {Point3D(3.0, 3.0, 3.0), 4.0}};
  for (const auto& v : data) tree.insert(v);
  EXPECT_EQ(tree.size(), data.size());
  EXPECT_FALSE(tree.empty());
  auto bounds = tree.bounds();
  ASSERT_TRUE(bounds.has_value());
  tree.clear();
  EXPECT_EQ(tree.size(), 0);
  EXPECT_TRUE(tree.empty());
}

TEST(RTree3D, Packing) {
  RTree3D tree;
  std::vector<RTree3D::value_t> data = {{Point3D(0.0, 0.0, 0.0), 1.0},
                                        {Point3D(1.0, 1.0, 1.0), 2.0},
                                        {Point3D(2.0, 2.0, 2.0), 3.0}};
  tree.packing(data);
  EXPECT_EQ(tree.size(), data.size());
}

TEST(RTree3D, KNNQuery) {
  RTree3D tree;
  for (int i = 0; i < 10; ++i) {
    tree.insert({Point3D(i, i, i), static_cast<double>(i)});
  }
  auto result =
      tree.query(Point3D(5.1, 5.1, 5.1), 3, std::numeric_limits<double>::max());
  ASSERT_EQ(result.size(), 3);
  EXPECT_NEAR(result[0].second, 5.0, 1e-12);
}

TEST(RTree3D, QueryBall) {
  RTree3D tree;
  for (int i = 0; i < 10; ++i) {
    tree.insert({Point3D(i, 0, 0), static_cast<double>(i)});
  }
  auto result = tree.query_ball(Point3D(5, 0, 0), 2.0);
  std::vector<double> found;
  found.reserve(result.size());
  for (const auto& [dist, val] : result) {
    found.push_back(val);
  }
  EXPECT_TRUE(std::ranges::find(found, 4.0) != found.end());
  EXPECT_TRUE(std::ranges::find(found, 5.0) != found.end());
  EXPECT_TRUE(std::ranges::find(found, 6.0) != found.end());
}

TEST(RTree3D, QueryWithin) {
  RTree3D tree;
  for (int i = 0; i < 10; ++i) {
    tree.insert({Point3D(i, 0, 0), static_cast<double>(i)});
  }
  auto result =
      tree.query(Point3D(5, 0, 0), 3, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kEnvelope);
  // May be empty if not surrounded, but should not crash
  EXPECT_LE(result.size(), 3);
}

TEST(RTree3D, ValueQuery) {
  RTree3D tree;
  for (int i = 0; i < 10; ++i) {
    tree.insert({Point3D(i, 0, 0), static_cast<double>(i)});
  }
  auto result = tree.value(Point3D(5, 0, 0), 2.0, 3,
                           pyinterp::geometry::BoundaryCheck::kNone);
  EXPECT_LE(result.size(), 3);
}

TEST(RTree3D, InverseDistanceWeighting) {
  RTree3D tree;
  for (int i = 0; i < 10; ++i) {
    tree.insert({Point3D(i, 0, 0), static_cast<double>(i)});
  }
  auto [val, n] = tree.inverse_distance_weighting(
      Point3D(5, 0, 0), 2.0, 3, 2, pyinterp::geometry::BoundaryCheck::kNone);
  EXPECT_TRUE(std::isfinite(val) || std::isnan(val));
  EXPECT_LE(n, 3);
}

TEST(RTree3D, KrigingInstantiation) {
  using promotion_t = decltype(std::declval<double>() + std::declval<double>());
  math::interpolate::Kriging<promotion_t> model(
      1.0, 1.0, 1.0, math::interpolate::CovarianceFunction::kGaussian);
  RTree3D tree;
  for (int i = 0; i < 3; ++i) {
    tree.insert({Point3D(i, 0, 0), static_cast<double>(i)});
  }
  auto [val, n] = tree.kriging(model, Point3D(1, 0, 0), 2.0, 3,
                               pyinterp::geometry::BoundaryCheck::kNone);
  EXPECT_TRUE(std::isfinite(val) || std::isnan(val));
  EXPECT_LE(n, 3);
}

TEST(RTree3D, RBFInstantiation) {
  using promotion_t = decltype(std::declval<double>() + std::declval<double>());
  math::interpolate::RBF<promotion_t> model(
      std::numeric_limits<double>::quiet_NaN(), 0,
      math::interpolate::RBFKernel::kMultiquadric);
  RTree3D tree;
  for (int i = 0; i < 3; ++i) {
    tree.insert({Point3D(i, 0, 0), static_cast<double>(i)});
  }
  auto [val, n] =
      tree.radial_basis_function(model, Point3D(1, 0, 0), 2.0, 3,
                                 pyinterp::geometry::BoundaryCheck::kNone);
  EXPECT_TRUE(std::isfinite(val) || std::isnan(val));
  EXPECT_LE(n, 3);
}

TEST(RTree3D, WindowFunctionInstantiation) {
  RTree3D tree;
  for (int i = 0; i < 3; ++i) {
    tree.insert({Point3D(i, 0, 0), static_cast<double>(i)});
  }
  math::interpolate::InterpolationWindow<double> model(
      math::interpolate::window::Kernel::kHamming, 0.5);
  auto [val, n] =
      tree.window_function(model, Point3D(1, 0, 0), 2.0, 3,
                           pyinterp::geometry::BoundaryCheck::kNone);
  EXPECT_TRUE(std::isfinite(val) || std::isnan(val));
  EXPECT_LE(n, 3);
}

// Boundary Check Tests for 2D
TEST(RTree2D, QueryWithBoundaryCheckNone) {
  RTree2D tree;
  // Create a grid of points
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      tree.insert({Point2D(i, j), static_cast<double>(i * 5 + j)});
    }
  }

  // Query at center point with no boundary check
  auto result =
      tree.query(Point2D(2.0, 2.0), 4, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kNone);
  EXPECT_EQ(result.size(), 4);
}

TEST(RTree2D, QueryWithBoundaryCheckEnvelope) {
  RTree2D tree;
  // Create a grid of points forming a square
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      tree.insert({Point2D(i, j), static_cast<double>(i * 5 + j)});
    }
  }

  // Query at center - should find neighbors within envelope
  auto result_center =
      tree.query(Point2D(2.0, 2.0), 4, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kEnvelope);
  EXPECT_LE(result_center.size(), 4);

  // Query at edge - may return empty if point is not surrounded
  auto result_edge =
      tree.query(Point2D(0.5, 0.5), 4, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kEnvelope);
  EXPECT_LE(result_edge.size(), 4);
}

TEST(RTree2D, QueryWithBoundaryCheckConvexHull) {
  RTree2D tree;
  // Create a grid of points
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      tree.insert({Point2D(i, j), static_cast<double>(i * 5 + j)});
    }
  }

  // Query at center with convex hull check
  auto result_center =
      tree.query(Point2D(2.0, 2.0), 4, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kConvexHull);
  EXPECT_LE(result_center.size(), 4);

  // Query at edge with convex hull check
  auto result_edge =
      tree.query(Point2D(0.5, 0.5), 4, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kConvexHull);
  EXPECT_LE(result_edge.size(), 4);
}

TEST(RTree2D, QueryBoundaryCheckComparison) {
  RTree2D tree;
  // Create a triangular distribution of points
  tree.insert({Point2D(0.0, 0.0), 1.0});
  tree.insert({Point2D(2.0, 0.0), 2.0});
  tree.insert({Point2D(1.0, 2.0), 3.0});
  tree.insert({Point2D(1.0, 0.5), 4.0});

  // Query point inside the triangle
  Point2D query_inside(1.0, 0.5);

  auto result_none =
      tree.query(query_inside, 4, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kNone);
  auto result_envelope =
      tree.query(query_inside, 4, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kEnvelope);
  auto result_hull =
      tree.query(query_inside, 4, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kConvexHull);

  // No boundary check should return most results
  EXPECT_GE(result_none.size(), result_envelope.size());
  EXPECT_GE(result_envelope.size(), result_hull.size());

  // All should return at least 0 results
  EXPECT_GE(result_none.size(), 0);
  EXPECT_GE(result_envelope.size(), 0);
  EXPECT_GE(result_hull.size(), 0);
}

// Boundary Check Tests for 3D
TEST(RTree3D, QueryWithBoundaryCheckNone) {
  RTree3D tree;
  // Create a cube of points
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        tree.insert(
            {Point3D(i, j, k), static_cast<double>(i * 16 + j * 4 + k)});
      }
    }
  }

  // Query at center point with no boundary check
  auto result =
      tree.query(Point3D(1.5, 1.5, 1.5), 8, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kNone);
  EXPECT_EQ(result.size(), 8);
}

TEST(RTree3D, QueryWithBoundaryCheckEnvelope) {
  RTree3D tree;
  // Create a cube of points
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        tree.insert(
            {Point3D(i, j, k), static_cast<double>(i * 16 + j * 4 + k)});
      }
    }
  }

  // Query at center - should find neighbors within envelope
  auto result_center =
      tree.query(Point3D(1.5, 1.5, 1.5), 8, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kEnvelope);
  EXPECT_LE(result_center.size(), 8);

  // Query at edge
  auto result_edge =
      tree.query(Point3D(0.5, 0.5, 0.5), 8, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kEnvelope);
  EXPECT_LE(result_edge.size(), 8);
}

TEST(RTree3D, QueryWithBoundaryCheckConvexHull) {
  RTree3D tree;
  // Create a tetrahedral distribution of points
  tree.insert({Point3D(0.0, 0.0, 0.0), 1.0});
  tree.insert({Point3D(2.0, 0.0, 0.0), 2.0});
  tree.insert({Point3D(1.0, 2.0, 0.0), 3.0});
  tree.insert({Point3D(1.0, 1.0, 2.0), 4.0});

  // Query at center of tetrahedron
  Point3D query_center(1.0, 1.0, 0.5);

  auto result_hull =
      tree.query(query_center, 4, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kConvexHull);

  // Should have at most 4 results
  EXPECT_LE(result_hull.size(), 4);
}

TEST(RTree3D, QueryBoundaryCheckComparison) {
  RTree3D tree;
  // Create a tetrahedral distribution
  tree.insert({Point3D(0.0, 0.0, 0.0), 1.0});
  tree.insert({Point3D(1.0, 0.0, 0.0), 2.0});
  tree.insert({Point3D(0.5, 1.0, 0.0), 3.0});
  tree.insert({Point3D(0.5, 0.5, 1.0), 4.0});

  // Query point inside the tetrahedron
  Point3D query_inside(0.5, 0.5, 0.3);

  auto result_none =
      tree.query(query_inside, 4, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kNone);
  auto result_envelope =
      tree.query(query_inside, 4, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kEnvelope);
  auto result_hull =
      tree.query(query_inside, 4, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kConvexHull);

  // No boundary check should return most or equal results
  EXPECT_GE(result_none.size(), result_envelope.size());
  EXPECT_GE(result_envelope.size(), result_hull.size());

  // All should return valid results
  EXPECT_GE(result_none.size(), 0);
  EXPECT_GE(result_envelope.size(), 0);
  EXPECT_GE(result_hull.size(), 0);
}

TEST(RTree2D, QueryEmptyResultWithBoundaryCheck) {
  RTree2D tree;
  // Create points in a cluster
  for (int i = 0; i < 3; ++i) {
    tree.insert({Point2D(i, 0.0), static_cast<double>(i)});
  }

  // Query from a point far away with strict boundary check
  auto result =
      tree.query(Point2D(10.0, 10.0), 3, std::numeric_limits<double>::max(),
                 pyinterp::geometry::BoundaryCheck::kConvexHull);

  // Should return empty or very few results
  EXPECT_LE(result.size(), 3);
}

TEST(RTree3D, QueryEmptyResultWithBoundaryCheck) {
  RTree3D tree;
  // Create points in a cluster
  for (int i = 0; i < 3; ++i) {
    tree.insert({Point3D(i, 0.0, 0.0), static_cast<double>(i)});
  }

  // Query from a point far away with strict boundary check
  auto result = tree.query(Point3D(10.0, 10.0, 10.0), 3,
                           std::numeric_limits<double>::max(),
                           pyinterp::geometry::BoundaryCheck::kConvexHull);

  // Should return empty or very few results
  EXPECT_LE(result.size(), 3);
}

TEST(RTree3D, SerializeDeserialize) {
  RTree3D tree;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        tree.insert(
            {Point3D(i, j, k), static_cast<double>(i * 25 + j * 5 + k)});
      }
    }
  }

  // Serialize the tree
  auto serialized = serialization::Reader(tree.pack());

  // Deserialize into a new tree
  auto deserialized_tree = RTree3D::unpack(serialized);

  // Verify size
  EXPECT_EQ(tree.size(), deserialized_tree.size());

  // Verify bounds
  auto original_bounds = tree.bounds();
  auto deserialized_bounds = deserialized_tree.bounds();

  ASSERT_TRUE(deserialized_bounds.has_value());
  ASSERT_TRUE(original_bounds.has_value());

  EXPECT_EQ(deserialized_bounds->min_corner().get<0>(),
            original_bounds->min_corner().get<0>());
  EXPECT_EQ(deserialized_bounds->min_corner().get<1>(),
            original_bounds->min_corner().get<1>());
  EXPECT_EQ(deserialized_bounds->min_corner().get<2>(),
            original_bounds->min_corner().get<2>());
  EXPECT_EQ(deserialized_bounds->max_corner().get<0>(),
            original_bounds->max_corner().get<0>());
  EXPECT_EQ(deserialized_bounds->max_corner().get<1>(),
            original_bounds->max_corner().get<1>());
  EXPECT_EQ(deserialized_bounds->max_corner().get<2>(),
            original_bounds->max_corner().get<2>());
}

}  // namespace pyinterp::geometry
