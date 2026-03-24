#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "pyinterp/geometry/satellite/algorithms/track_decomposition.hpp"

namespace pyinterp::geometry::satellite::algorithms {

// ---------------------------------------------------------------------------
// Helpers to generate synthetic tracks
// ---------------------------------------------------------------------------

/// Generate a monotonic latitude sequence from `start` to `end` (inclusive)
/// with `n` points, and a corresponding simple longitude sequence.
struct SyntheticTrack {
  std::vector<double> lon;
  std::vector<double> lat;
};

auto make_linear_track(double lat_start, double lat_end, size_t n,
                       double lon_start = 0.0, double lon_rate = 0.1)
    -> SyntheticTrack {
  SyntheticTrack t;
  t.lon.resize(n);
  t.lat.resize(n);
  for (size_t i = 0; i < n; ++i) {
    auto const frac = (n > 1) ? static_cast<double>(i) / (n - 1) : 0.0;
    t.lat[i] = lat_start + frac * (lat_end - lat_start);
    t.lon[i] = lon_start + i * lon_rate;
  }
  return t;
}

/// Generate a V-shaped or Λ-shaped track (up then down or down then up).
auto make_v_track(double lat_start, double lat_peak, double lat_end,
                  size_t n_up, size_t n_down, double lon_start = 0.0,
                  double lon_rate = 0.1) -> SyntheticTrack {
  auto up = make_linear_track(lat_start, lat_peak, n_up, lon_start, lon_rate);
  auto down = make_linear_track(lat_peak, lat_end, n_down,
                                lon_start + n_up * lon_rate, lon_rate);
  // Remove duplicate peak point.
  SyntheticTrack t;
  t.lon.insert(t.lon.end(), up.lon.begin(), up.lon.end());
  t.lon.insert(t.lon.end(), down.lon.begin() + 1, down.lon.end());
  t.lat.insert(t.lat.end(), up.lat.begin(), up.lat.end());
  t.lat.insert(t.lat.end(), down.lat.begin() + 1, down.lat.end());
  return t;
}

// ===================================================================
// Orbit direction inference
// ===================================================================

TEST(OrbitDirection, ProgradeLongitude) {
  // Increasing longitude -> prograde.
  std::vector<double> lon(20);
  // NOLINTNEXTLINE(modernize-use-ranges, boost-use-ranges)
  std::iota(lon.begin(), lon.end(), 0.0);
  EXPECT_EQ(infer_orbit_direction(
                Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size())),
            OrbitDirection::kPrograde);
}

TEST(OrbitDirection, RetrogradeLongitude) {
  // Decreasing longitude -> retrograde.
  std::vector<double> lon(20);
  for (size_t i = 0; i < lon.size(); ++i) {
    lon[i] = 100.0 - static_cast<double>(i);
  }
  EXPECT_EQ(infer_orbit_direction(
                Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size())),
            OrbitDirection::kRetrograde);
}

TEST(OrbitDirection, TieResolvesToRetrograde) {
  // Alternating -> tie -> retrograde.
  std::vector<double> lon = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  EXPECT_EQ(infer_orbit_direction(
                Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size())),
            OrbitDirection::kRetrograde);
}

TEST(OrbitDirection, SinglePoint) {
  std::vector<double> lon = {42.0};
  EXPECT_EQ(infer_orbit_direction(
                Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size())),
            OrbitDirection::kRetrograde);
}

// ===================================================================
// Latitude classification
// ===================================================================

TEST(ClassifyLatitude, BoundaryInclusion) {
  EXPECT_EQ(classify_latitude(-50.0, -50.0, 50.0), LatitudeZone::kSouth);
  EXPECT_EQ(classify_latitude(50.0, -50.0, 50.0), LatitudeZone::kNorth);
  EXPECT_EQ(classify_latitude(-49.999, -50.0, 50.0), LatitudeZone::kMid);
  EXPECT_EQ(classify_latitude(49.999, -50.0, 50.0), LatitudeZone::kMid);
}

// ===================================================================
// Latitude bands strategy — test vectors from spec
// ===================================================================

TEST(LatitudeBands, Blocks3NoMerge) {
  // Monotonic -90 -> +90, limits (-50, +50).  Expect 3 blocks.
  auto [lon, lat] = make_linear_track(-90.0, 90.0, 181);

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands);

  ASSERT_EQ(segments.size(), 3u);
  EXPECT_EQ(segments[0].zone, LatitudeZone::kSouth);
  EXPECT_EQ(segments[1].zone, LatitudeZone::kMid);
  EXPECT_EQ(segments[2].zone, LatitudeZone::kNorth);

  // Contiguity.
  EXPECT_EQ(segments[0].first_index, 0u);
  for (size_t i = 1; i < segments.size(); ++i) {
    EXPECT_EQ(segments[i].first_index, segments[i - 1].last_index + 1);
  }
  EXPECT_EQ(segments.back().last_index, 180u);
}

TEST(LatitudeBands, Blocks2NoMerge) {
  // Monotonic -90 -> +49.5, limits (-50, +50).  Expect 2 blocks: SOUTH, MID.
  auto [lon, lat] = make_linear_track(-90.0, 49.5, 140);

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands);

  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0].zone, LatitudeZone::kSouth);
  EXPECT_EQ(segments[1].zone, LatitudeZone::kMid);
}

TEST(LatitudeBands, Blocks5NoMerge) {
  // Up-down: -90 -> +90 -> -90.  Expect 5 blocks: S, M, N, M, S.
  auto [lon, lat] = make_v_track(-90.0, 90.0, -90.0, 181, 181);

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands);

  ASSERT_EQ(segments.size(), 5u);
  EXPECT_EQ(segments[0].zone, LatitudeZone::kSouth);
  EXPECT_EQ(segments[1].zone, LatitudeZone::kMid);
  EXPECT_EQ(segments[2].zone, LatitudeZone::kNorth);
  EXPECT_EQ(segments[3].zone, LatitudeZone::kMid);
  EXPECT_EQ(segments[4].zone, LatitudeZone::kSouth);
}

// ===================================================================
// Edge-merge policy
// ===================================================================

TEST(EdgeMerge, K2SmallFirst) {
  // Two blocks, first is small -> merge into one.
  // Latitude: 3 points south, 50 points mid.
  std::vector<double> lat(53);
  std::fill_n(lat.begin(), 3, -60.0);
  std::fill_n(lat.begin() + 3, 50, 0.0);
  std::vector<double> lon(53);
  // NOLINTNEXTLINE(modernize-use-ranges, boost-use-ranges)
  std::iota(lon.begin(), lon.end(), 0.0);

  auto opts = DecompositionOptions().with_min_edge_size(10);

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands, opts);

  EXPECT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0].first_index, 0u);
  EXPECT_EQ(segments[0].last_index, 52u);
}

TEST(EdgeMerge, K2SmallLast) {
  std::vector<double> lat(53);
  std::fill_n(lat.begin(), 50, 0.0);
  std::fill_n(lat.begin() + 50, 3, 60.0);
  std::vector<double> lon(53);
  // NOLINTNEXTLINE(modernize-use-ranges, boost-use-ranges)
  std::iota(lon.begin(), lon.end(), 0.0);

  auto opts = DecompositionOptions().with_min_edge_size(10);

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands, opts);

  EXPECT_EQ(segments.size(), 1u);
}

TEST(EdgeMerge, K2NeitherSmall) {
  std::vector<double> lat(100);
  std::fill_n(lat.begin(), 50, -60.0);
  std::fill_n(lat.begin() + 50, 50, 0.0);
  std::vector<double> lon(100);
  // NOLINTNEXTLINE(modernize-use-ranges, boost-use-ranges)
  std::iota(lon.begin(), lon.end(), 0.0);

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands);

  EXPECT_EQ(segments.size(), 2u);
}

TEST(EdgeMerge, K3SmallBoth) {
  // S(3), M(50), N(3) -> merge all.
  std::vector<double> lat;
  lat.insert(lat.end(), 3, -60.0);
  lat.insert(lat.end(), 50, 0.0);
  lat.insert(lat.end(), 3, 60.0);
  std::vector<double> lon(lat.size());
  // NOLINTNEXTLINE(modernize-use-ranges, boost-use-ranges)
  std::iota(lon.begin(), lon.end(), 0.0);

  auto opts = DecompositionOptions().with_min_edge_size(10);

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands, opts);

  EXPECT_EQ(segments.size(), 1u);
}

TEST(EdgeMerge, K3SmallFirstOnly) {
  // S(3), M(50), N(20) -> merge first two, keep last.
  std::vector<double> lat;
  lat.insert(lat.end(), 3, -60.0);
  lat.insert(lat.end(), 50, 0.0);
  lat.insert(lat.end(), 20, 60.0);
  std::vector<double> lon(lat.size());
  // NOLINTNEXTLINE(modernize-use-ranges, boost-use-ranges)
  std::iota(lon.begin(), lon.end(), 0.0);

  auto opts = DecompositionOptions().with_min_edge_size(10);

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands, opts);

  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0].first_index, 0u);
  EXPECT_EQ(segments[0].last_index, 52u);
  EXPECT_EQ(segments[1].first_index, 53u);
}

TEST(EdgeMerge, K3SmallLastOnly) {
  // S(20), M(50), N(3) -> keep first, merge last two.
  std::vector<double> lat;
  lat.insert(lat.end(), 20, -60.0);
  lat.insert(lat.end(), 50, 0.0);
  lat.insert(lat.end(), 3, 60.0);
  std::vector<double> lon(lat.size());
  // NOLINTNEXTLINE(modernize-use-ranges, boost-use-ranges)
  std::iota(lon.begin(), lon.end(), 0.0);

  auto opts = DecompositionOptions().with_min_edge_size(10);

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands, opts);

  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[1].last_index, lat.size() - 1);
}

TEST(EdgeMerge, K4SmallBoth) {
  // S(3), M(50), N(50), M_end(3) -> merge edges.
  // Build a latitude: south(3), mid(50), north(50), another_zone(3).
  // To get 4 blocks we need 4 zone transitions.
  std::vector<double> lat;
  lat.insert(lat.end(), 3, -60.0);
  lat.insert(lat.end(), 50, 0.0);
  lat.insert(lat.end(), 50, 60.0);
  lat.insert(lat.end(), 3, 0.0);
  std::vector<double> lon(lat.size());
  // NOLINTNEXTLINE(modernize-use-ranges, boost-use-ranges)
  std::iota(lon.begin(), lon.end(), 0.0);

  auto opts = DecompositionOptions().with_min_edge_size(10);

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands, opts);

  // First and last merged: (0,1) merged, (2,3) merged -> 2 blocks.
  EXPECT_EQ(segments.size(), 2u);
}

// ===================================================================
// Monotonic segments strategy
// ===================================================================

TEST(MonotonicSegments, SimpleAscending) {
  // Purely ascending latitude -> 1 segment.
  auto [lon, lat] = make_linear_track(-70.0, 70.0, 200);

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kMonotonicSegments);

  EXPECT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0].first_index, 0u);
  EXPECT_EQ(segments[0].last_index, 199u);
}

TEST(MonotonicSegments, VShape) {
  // Up then down -> 2 monotonic segments (overlapping by 1 at peak).
  auto [lon, lat] = make_v_track(-70.0, 70.0, -70.0, 100, 100);

  auto opts =
      DecompositionOptions().with_min_edge_size(1).with_merge_area_ratio(
          0.0);  // Disable area merge.

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kMonotonicSegments, opts);

  ASSERT_EQ(segments.size(), 2u);
  // The peak point should be shared (last of first = first of second).
  EXPECT_EQ(segments[0].last_index, segments[1].first_index);
}

TEST(MonotonicSegments, MultipleReversals) {
  // Zigzag: up-down-up-down.
  auto t1 = make_linear_track(-70.0, 70.0, 80);
  auto t2 = make_linear_track(70.0, -30.0, 60, 80 * 0.1, 0.1);
  auto t3 = make_linear_track(-30.0, 50.0, 50, 140 * 0.1, 0.1);

  SyntheticTrack t;
  auto append = [&](SyntheticTrack const& src, bool skip_first) {
    auto start = skip_first ? 1u : 0u;
    t.lon.insert(t.lon.end(), src.lon.begin() + start, src.lon.end());
    t.lat.insert(t.lat.end(), src.lat.begin() + start, src.lat.end());
  };
  append(t1, false);
  append(t2, true);
  append(t3, true);

  auto opts =
      DecompositionOptions().with_min_edge_size(1).with_merge_area_ratio(
          0.0);  // Disable area merge.

  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(t.lon.data(), t.lon.size()),
                      Eigen::Map<Eigen::VectorXd>(t.lat.data(), t.lat.size()),
                      DecompositionStrategy::kMonotonicSegments, opts);

  // 3 monotonic runs: up, down, up.
  EXPECT_EQ(segments.size(), 3u);
}

// ===================================================================
// Swath-width expansion
// ===================================================================

TEST(SwathExpansion, WidensAtHighLatitude) {
  // A track at high latitude should have wider longitude expansion.
  auto [lon, lat] = make_linear_track(70.0, 80.0, 50);

  auto opts_no_swath = DecompositionOptions().with_swath_width_km(0.0);
  auto opts_swath = DecompositionOptions().with_swath_width_km(140.0);

  auto seg_no =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands, opts_no_swath);
  auto seg_sw =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands, opts_swath);

  ASSERT_EQ(seg_no.size(), 1u);
  ASSERT_EQ(seg_sw.size(), 1u);

  // Swath version should be wider in both lon and lat.
  EXPECT_LT(seg_sw[0].bbox.min_corner().lon(),
            seg_no[0].bbox.min_corner().lon());
  EXPECT_GT(seg_sw[0].bbox.max_corner().lon(),
            seg_no[0].bbox.max_corner().lon());
  EXPECT_LT(seg_sw[0].bbox.min_corner().lat(),
            seg_no[0].bbox.min_corner().lat());
  EXPECT_GT(seg_sw[0].bbox.max_corner().lat(),
            seg_no[0].bbox.max_corner().lat());
}

TEST(SwathExpansion, LargerAtPolesThanEquator) {
  // Compare longitude expansion at equator vs. at 70° latitude.
  auto [lon_eq, lat_eq] = make_linear_track(-5.0, 5.0, 50);
  auto [lon_hi, lat_hi] = make_linear_track(65.0, 75.0, 50);

  auto opts = DecompositionOptions().with_swath_width_km(140.0);

  auto seg_eq =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon_eq.data(), lon_eq.size()),
                      Eigen::Map<Eigen::VectorXd>(lat_eq.data(), lat_eq.size()),
                      DecompositionStrategy::kLatitudeBands, opts);
  auto seg_hi =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon_hi.data(), lon_hi.size()),
                      Eigen::Map<Eigen::VectorXd>(lat_hi.data(), lat_hi.size()),
                      DecompositionStrategy::kLatitudeBands, opts);

  auto lon_span_eq =
      seg_eq[0].bbox.max_corner().lon() - seg_eq[0].bbox.min_corner().lon();
  auto lon_span_hi =
      seg_hi[0].bbox.max_corner().lon() - seg_hi[0].bbox.min_corner().lon();

  // High-latitude segment should have a larger longitude span due to 1/cos(φ).
  EXPECT_GT(lon_span_hi, lon_span_eq);
}

TEST(SwathExpansion, ZeroSwathMatchesNadir) {
  auto [lon, lat] = make_linear_track(-30.0, 30.0, 100);

  auto opts_zero = DecompositionOptions().with_swath_width_km(0.0);

  auto seg =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands, opts_zero);

  // Bbox should exactly match the data extent (no expansion).
  EXPECT_DOUBLE_EQ(seg[0].bbox.min_corner().lat(), -30.0);
  EXPECT_DOUBLE_EQ(seg[0].bbox.max_corner().lat(), 30.0);
}

// ===================================================================
// Coverage and contiguity postconditions
// ===================================================================

TEST(Postconditions, CoverageAndContiguity) {
  // For any decomposition, verify:
  //   - first segment starts at 0
  //   - last segment ends at N-1
  //   - consecutive segments are contiguous (next.first = prev.last + 1)
  //     OR overlap by exactly 1 (monotonic strategy)
  auto [lon, lat] = make_v_track(-90.0, 90.0, -90.0, 181, 181);

  for (auto strategy : {DecompositionStrategy::kLatitudeBands,
                        DecompositionStrategy::kMonotonicSegments}) {
    auto opts = DecompositionOptions().with_merge_area_ratio(
        0.0);  // Disable area merge for cleaner test.

    auto segments = decompose_track(
        Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
        Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()), strategy, opts);

    ASSERT_FALSE(segments.empty());
    EXPECT_EQ(segments.front().first_index, 0u);
    EXPECT_EQ(segments.back().last_index, lon.size() - 1);

    for (size_t i = 1; i < segments.size(); ++i) {
      auto gap = static_cast<int64_t>(segments[i].first_index) -
                 static_cast<int64_t>(segments[i - 1].last_index);
      // gap should be 0 (overlap of 1, monotonic) or 1 (contiguous, bands).
      EXPECT_TRUE(gap == 0 || gap == 1)
          << "Gap between segment " << i - 1 << " and " << i << " is " << gap;
    }
  }
}

// ===================================================================
// filter_by_extent
// ===================================================================

TEST(FilterByExtent, DropsNonIntersecting) {
  auto [lon, lat] = make_linear_track(-90.0, 90.0, 181);
  auto segments =
      decompose_track(Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                      Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                      DecompositionStrategy::kLatitudeBands);

  // Extent covering only the mid-latitude range.
  BoundingBox extent{{-180.0, -30.0}, {180.0, 30.0}};

  auto filtered = filter_by_extent(segments, extent);

  // Should keep only the MID segment (and possibly edges that overlap).
  // At minimum, the MID segment must be present.
  bool has_mid = false;
  for (auto const& s : filtered) {
    if (s.zone == LatitudeZone::kMid) {
      has_mid = true;
    }
  }
  EXPECT_TRUE(has_mid);

  // Pure SOUTH segment (lat <= -50) should NOT intersect [-30, 30].
  for (auto const& s : filtered) {
    // The SOUTH segment's bbox.lat_max should be <= -50, which doesn't
    // overlap [-30, 30].  But with default limits (-50, 50), the south
    // segment goes down to -90 and up to -50.  -50 < -30, so no overlap.
    if (s.zone == LatitudeZone::kSouth) {
      EXPECT_GT(s.bbox.max_corner().lat(), extent.min_corner().lat())
          << "SOUTH segment shouldn't be here if it doesn't overlap";
    }
  }
}

// ===================================================================
// Error handling
// ===================================================================

TEST(ErrorHandling, EmptyInput) {
  std::vector<double> lon, lat;
  EXPECT_THROW(static_cast<void>(decompose_track(
                   Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                   Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                   DecompositionStrategy::kLatitudeBands)),
               std::invalid_argument);
}

TEST(ErrorHandling, SizeMismatch) {
  std::vector<double> lon = {1, 2, 3};
  std::vector<double> lat = {1, 2};
  EXPECT_THROW(static_cast<void>(decompose_track(
                   Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                   Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                   DecompositionStrategy::kLatitudeBands)),
               std::invalid_argument);
}

TEST(ErrorHandling, InvalidLimits) {
  auto [lon, lat] = make_linear_track(-90.0, 90.0, 100);
  auto opts = DecompositionOptions().with_south_limit(50.0).with_north_limit(
      -50.0);  // Inverted!
  EXPECT_THROW(static_cast<void>(decompose_track(
                   Eigen::Map<Eigen::VectorXd>(lon.data(), lon.size()),
                   Eigen::Map<Eigen::VectorXd>(lat.data(), lat.size()),
                   DecompositionStrategy::kLatitudeBands, opts)),
               std::invalid_argument);
}

}  // namespace pyinterp::geometry::satellite::algorithms
