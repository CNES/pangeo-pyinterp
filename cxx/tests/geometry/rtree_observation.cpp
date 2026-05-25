// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pyinterp/geometry/rtree.hpp"
#include "pyinterp/geometry/rtree_value_traits.hpp"
#include "pyinterp/math/interpolate/observation.hpp"
#include "pyinterp/serialization_buffer.hpp"

using Point3D =
    boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian>;

namespace pyinterp::geometry {

using Obs = math::interpolate::Observation<double>;
using RTreeObs = RTree<Point3D, Obs>;

// The Observation specialization must contribute a non-zero serialization
// tag — otherwise its pickles would collide with scalar trees of the same
// dimension.
TEST(Observation, SerializationTagDiffersFromScalar) {
  static_assert(value_traits<double>::serialization_tag == 0,
                "Scalar tag must remain 0 for backward compatibility");
  static_assert(value_traits<Obs>::serialization_tag != 0,
                "Observation must have a distinct serialization tag");
}

TEST(RTreeObservation, PackingAndSize) {
  RTreeObs tree;
  std::vector<RTreeObs::value_t> data = {
      {Point3D(0.0, 0.0, 0.0), {.value = 1.0, .sigma2 = 0.01}},
      {Point3D(1.0, 0.0, 0.0), {.value = 2.0, .sigma2 = 0.04}},
      {Point3D(0.0, 1.0, 0.0), {.value = 3.0, .sigma2 = 0.09}},
      {Point3D(1.0, 1.0, 0.0), {.value = 4.0, .sigma2 = 0.16}},
  };
  tree.packing(data);
  EXPECT_EQ(tree.size(), data.size());
  EXPECT_FALSE(tree.empty());
}

// k-NN query on an Observation tree returns the right Observation values
// alongside distances, exactly like the scalar case.
TEST(RTreeObservation, QueryReturnsObservations) {
  RTreeObs tree;
  std::vector<RTreeObs::value_t> data = {
      {Point3D(0.0, 0.0, 0.0), {.value = 10.0, .sigma2 = 0.1}},
      {Point3D(5.0, 0.0, 0.0), {.value = 20.0, .sigma2 = 0.2}},
      {Point3D(0.0, 5.0, 0.0), {.value = 30.0, .sigma2 = 0.3}},
  };
  tree.packing(data);

  auto results = tree.query(Point3D(0.1, 0.1, 0.0), 2,
                            std::numeric_limits<double>::max(),
                            BoundaryCheck::kNone);
  ASSERT_EQ(results.size(), 2U);
  // Closest is (0,0,0).
  EXPECT_DOUBLE_EQ(results[0].second.value, 10.0);
  EXPECT_DOUBLE_EQ(results[0].second.sigma2, 0.1);
}

// Serialization round-trip must preserve every (point, observation) record.
TEST(RTreeObservation, SerializationRoundTrip) {
  RTreeObs tree;
  std::vector<RTreeObs::value_t> data = {
      {Point3D(0.0, 0.0, 0.0), {.value = 1.5, .sigma2 = 0.01}},
      {Point3D(2.5, -1.0, 3.0), {.value = -7.25, .sigma2 = 4.0}},
      {Point3D(10.0, 10.0, 10.0), {.value = 0.0, .sigma2 = 1e-8}},
  };
  tree.packing(data);

  auto writer = tree.pack();
  auto reader = serialization::Reader(writer.data(), writer.size());
  auto restored = RTreeObs::unpack(reader);

  EXPECT_EQ(restored.size(), tree.size());

  // Validate each round-tripped value.
  auto sorted = [](auto& t) {
    std::vector<std::pair<Point3D, Obs>> v;
    v.reserve(t.size());
    auto results = t.query(Point3D(0.0, 0.0, 0.0), 10,
                           std::numeric_limits<double>::max(),
                           BoundaryCheck::kNone);
    return results;
  };
  auto original = sorted(tree);
  auto roundtrip = sorted(restored);
  ASSERT_EQ(original.size(), roundtrip.size());
  for (size_t i = 0; i < original.size(); ++i) {
    EXPECT_DOUBLE_EQ(original[i].first, roundtrip[i].first);
    EXPECT_EQ(original[i].second, roundtrip[i].second);
  }
}

// An Observation pickle must be rejected when loaded as a scalar tree (their
// magic numbers differ thanks to the trait tag). This guarantees we never
// silently misinterpret bytes.
TEST(RTreeObservation, CrossLoadingRejected) {
  RTreeObs obs_tree;
  obs_tree.packing({{Point3D(0.0, 0.0, 0.0), {.value = 1.0, .sigma2 = 0.5}}});
  auto writer = obs_tree.pack();
  auto reader = serialization::Reader(writer.data(), writer.size());

  using ScalarTree = RTree<Point3D, double>;
  EXPECT_THROW(ScalarTree::unpack(reader), std::invalid_argument);
}

// Conversely, the scalar serialization layout is unchanged — existing
// pickles produced by older versions must still load.
TEST(RTreeObservation, ScalarMagicNumberUnchanged) {
  using ScalarTree = RTree<Point3D, double>;
  // The scalar magic number must equal the pre-refactor value (`'RTTR' + 3`).
  static_assert(0x52545452 + 3 ==
                    0x52545452 + 3 + value_traits<double>::serialization_tag,
                "Scalar magic number must remain 0x52545455 in 3D");

  ScalarTree tree;
  tree.packing({{Point3D(1.0, 2.0, 3.0), 4.0},
                {Point3D(4.0, 5.0, 6.0), 7.0}});
  auto writer = tree.pack();
  auto reader = serialization::Reader(writer.data(), writer.size());
  auto restored = ScalarTree::unpack(reader);
  EXPECT_EQ(restored.size(), 2U);
}

}  // namespace pyinterp::geometry
