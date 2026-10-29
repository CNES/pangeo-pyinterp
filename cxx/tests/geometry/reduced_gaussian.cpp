// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/reduced_gaussian/grid.hpp"
#include "pyinterp/geometry/reduced_gaussian/interpolate.hpp"
#include "pyinterp/geometry/reduced_gaussian/mesh.hpp"
#include "pyinterp/geometry/reduced_gaussian/zone.hpp"

namespace pyinterp::geometry::reduced_gaussian {
namespace {

using Ranges = Eigen::Matrix<int64_t, Eigen::Dynamic, 2, Eigen::RowMajor>;

constexpr double kTolerance = 1e-12;

/// The N2 mesh of the c3po tests: 8, 16, 16 and 8 points.
auto make_n2_mesh() -> std::shared_ptr<const Mesh> {
  Vector<double> lats(4);
  lats << 54.7356, 19.4712, -19.4712, -54.7356;
  Vector<int64_t> pl(4);
  pl << 8, 16, 16, 8;
  return std::make_shared<const Mesh>(std::move(lats), std::move(pl));
}

/// A mesh whose parallels hold numbers of points that do not divide 360.
auto make_rounding_mesh() -> std::shared_ptr<const Mesh> {
  Vector<double> lats(4);
  lats << 54.7356, 19.4712, -19.4712, -54.7356;
  Vector<int64_t> pl(4);
  pl << 7, 19, 23, 13;
  return std::make_shared<const Mesh>(std::move(lats), std::move(pl));
}

/// The values of a function of the coordinates at the points of a mesh.
template <typename Function>
auto sample(const Mesh& mesh, Function&& function) -> std::vector<double> {
  std::vector<double> values(static_cast<size_t>(mesh.size()));
  for (int64_t jx = 0; jx < mesh.ny(); ++jx) {
    for (int64_t ix = 0; ix < mesh.points_per_parallel()[jx]; ++ix) {
      values[static_cast<size_t>(mesh.offsets()[jx] + ix)] =
          function(mesh.longitude(ix, jx), mesh.latitudes()[jx]);
    }
  }
  return values;
}

/// A field view of a vector of values.
auto field(const std::vector<double>& values, int64_t width = 1) -> Field {
  return {.name = "f",
          .data = values.data(),
          .rows = static_cast<int64_t>(values.size()) / width,
          .width = width,
          .dtype = DType::kFloat64,
          .has_levels = width > 1};
}

/// Interpolate the single field of a series at a point; NaN on failure.
auto interpolate(const Series& series, double lon, double lat,
                 std::optional<int64_t> time = std::nullopt,
                 BracketRule rule = {}) -> double {
  const auto* times = time ? &*time : nullptr;
  const Evaluator evaluator(series, {0}, times, rule);
  auto scratch = evaluator.make_scratch();
  std::vector<double> column(static_cast<size_t>(evaluator.total_width()));
  int64_t failed = 0;
  if (evaluator.evaluate(0, lon, lat, column.data(), scratch, failed) !=
      Status::kOk) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return column.front();
}

/// A series of one step holding a grid.
auto single(const std::shared_ptr<const Mesh>& mesh,
            const std::vector<double>& values,
            const std::shared_ptr<const Zone>& zone = nullptr) -> Series {
  Series series(mesh);
  series.append(
      0, std::make_shared<const Grid>(mesh, zone, std::vector{field(values)}));
  return series;
}

}  // namespace

TEST(ReducedGaussianMesh, Validation) {
  Vector<double> lats(2);
  lats << 10.0, -10.0;
  Vector<int64_t> pl(1);
  pl << 4;
  EXPECT_THROW(Mesh(lats, pl), std::invalid_argument);
  pl.resize(2);
  pl << 4, 0;
  EXPECT_THROW(Mesh(lats, pl), std::invalid_argument);
  pl << 4, 4;
  Vector<double> reversed(2);
  reversed << -10.0, 10.0;
  EXPECT_THROW(Mesh(reversed, pl), std::invalid_argument);
  const Mesh mesh(lats, pl);
  EXPECT_EQ(mesh.size(), 8);
  EXPECT_EQ(mesh.parallel_of(5), 1);
  EXPECT_TRUE(mesh == Mesh(lats, pl));
}

TEST(ReducedGaussianMesh, OutsideTheParallels) {
  const auto mesh = make_n2_mesh();
  Cell cell;
  EXPECT_FALSE(mesh->locate(10.0, 60.0, cell));
  EXPECT_FALSE(mesh->locate(10.0, -60.0, cell));
  EXPECT_FALSE(
      mesh->locate(std::numeric_limits<double>::quiet_NaN(), 0.0, cell));
  EXPECT_TRUE(mesh->locate(10.0, 54.7356, cell));
  EXPECT_TRUE(mesh->locate(10.0, -54.7356, cell));
}

TEST(ReducedGaussianMesh, WeightsBlendAcrossThePrimeMeridianSeam) {
  const auto mesh = make_n2_mesh();
  constexpr double kLat = 19.4712;  // parallel 1, 16 points, 22.5° cells
  constexpr double kStep = 22.5;
  Cell seam;
  Cell interior;
  for (const double fraction : {0.0, 0.25, 0.5, 0.9}) {
    const double lon = 337.5 + (fraction * kStep);
    ASSERT_TRUE(mesh->locate(lon, kLat, seam));
    EXPECT_NEAR(seam.weight[0], fraction, kTolerance);
    EXPECT_NEAR(seam.weight[1], 1.0 - fraction, kTolerance);
    ASSERT_TRUE(
        mesh->locate((4.0 * kStep) + (fraction * kStep), kLat, interior));
    for (size_t k = 0; k < kCorners; ++k) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
      EXPECT_NEAR(seam.weight[k], interior.weight[k], kTolerance);
    }
  }
}

TEST(ReducedGaussianMesh, SeamCellBetweenRowsMatchesAnInnerCell) {
  const auto mesh = make_n2_mesh();
  const double lat = 0.5 * (54.7356 + 19.4712);
  Cell seam;
  Cell inner;
  ASSERT_TRUE(mesh->locate(330.0, lat, seam));
  ASSERT_TRUE(mesh->locate(60.0, lat, inner));
  double sum = 0.0;
  for (size_t k = 0; k < kCorners; ++k) {
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
    EXPECT_NEAR(seam.weight[k], inner.weight[k], kTolerance);
    sum += seam.weight[k];
    // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
  }
  EXPECT_NEAR(sum, 1.0, kTolerance);
}

TEST(ReducedGaussianMesh, InterpolateAcrossTheSeamFollowsTheField) {
  const auto mesh = make_n2_mesh();
  const auto values = sample(*mesh, [](double lon, double) {
    return std::cos(lon * std::numbers::pi / 180.0);
  });
  const auto series = single(mesh, values);
  const double expected =
      0.5 * (std::cos(337.5 * std::numbers::pi / 180.0) + 1.0);
  EXPECT_NEAR(interpolate(series, 348.75, 19.4712), expected, kTolerance);
  // The same point in other turns.
  EXPECT_NEAR(interpolate(series, 348.75 - 360.0, 19.4712), expected,
              kTolerance);
  EXPECT_NEAR(interpolate(series, 348.75 + 720.0, 19.4712), expected,
              kTolerance);
}

TEST(ReducedGaussianMesh, AQueryAnUlpWestOfAKnotReadsThatKnot) {
  const auto mesh = make_rounding_mesh();
  const auto values = sample(
      *mesh, [](double lon, double lat) { return lon + (1000.0 * lat); });
  const auto series = single(mesh, values);
  for (int64_t jx = 0; jx < mesh->ny(); ++jx) {
    for (int64_t ix = 0; ix < mesh->points_per_parallel()[jx]; ++ix) {
      const auto knot = mesh->longitude(ix, jx);
      const auto lon =
          std::nextafter(knot, -std::numeric_limits<double>::infinity());
      const auto lat = mesh->latitudes()[jx];
      EXPECT_NEAR(interpolate(series, lon, lat),
                  values[static_cast<size_t>(mesh->offsets()[jx] + ix)], 1e-9)
          << jx << ", " << ix;
    }
  }
}

TEST(ReducedGaussianMesh, AQueryAtTheEndOfTheTurnReadsTheFirstKnot) {
  const auto mesh = make_rounding_mesh();
  const auto values = sample(*mesh, [](double lon, double lat) {
    return std::cos(lon * std::numbers::pi / 180.0) + lat;
  });
  const auto series = single(mesh, values);
  const double below = std::nextafter(360.0, 0.0);
  for (int64_t jx = 0; jx < mesh->ny(); ++jx) {
    const auto lat = mesh->latitudes()[jx];
    const auto first = values[static_cast<size_t>(mesh->offsets()[jx])];
    EXPECT_NEAR(interpolate(series, 360.0, lat), first, 1e-9);
    EXPECT_NEAR(interpolate(series, below, lat), first, 1e-9);
  }
  // Continuous across the seam between rows.
  for (int64_t jx = 0; jx + 1 < mesh->ny(); ++jx) {
    const double lat =
        0.5 * (mesh->latitudes()[jx] + mesh->latitudes()[jx + 1]);
    EXPECT_NEAR(interpolate(series, below, lat), interpolate(series, 0, lat),
                1e-9);
  }
}

TEST(ReducedGaussianMesh, WalksToTheBracketingParallel) {
  Vector<int64_t> pl(4);
  pl << 8, 16, 16, 8;
  Vector<double> north(4);
  north << 80.0, 78.0, 76.0, -80.0;
  const Mesh crowded_north(north, pl);
  Cell cell;
  ASSERT_TRUE(crowded_north.locate(10.0, 77.0, cell));
  EXPECT_EQ(crowded_north.parallel_of(cell.index[0]), 2);  // bottom
  EXPECT_EQ(crowded_north.parallel_of(cell.index[2]), 1);  // top
  Vector<double> south(4);
  south << 80.0, -76.0, -78.0, -80.0;
  const Mesh crowded_south(south, pl);
  ASSERT_TRUE(crowded_south.locate(10.0, 0.0, cell));
  EXPECT_EQ(crowded_south.parallel_of(cell.index[0]), 1);
  EXPECT_EQ(crowded_south.parallel_of(cell.index[2]), 0);
}

TEST(ReducedGaussianMesh, DegenerateCellSplitsEvenly) {
  const auto weights =
      Mesh::weights(10.0, 20.0, 20.0, 20.0, 30.0, 0.0, 30.0, 0.0);
  for (const auto weight : weights) {
    EXPECT_DOUBLE_EQ(weight, 0.25);
  }
}

TEST(ReducedGaussianCombine, NanCornersAreRenormalizedPerValue) {
  // Two points of width 2: the second value of the first point is missing.
  const std::vector<float> values{1.0F, std::numeric_limits<float>::quiet_NaN(),
                                  3.0F, 4.0F};
  const Field f{.name = "f",
                .data = values.data(),
                .rows = 2,
                .width = 2,
                .dtype = DType::kFloat32,
                .has_levels = true};
  std::array<double, 2> out{};
  std::array<double, 2> scratch{};
  combine(f, {0, 1, 0, 1}, {0.25, 0.25, 0.25, 0.25}, out.data(),
          scratch.data());
  EXPECT_DOUBLE_EQ(out[0], 2.0);
  EXPECT_DOUBLE_EQ(out[1], 4.0);  // only the second point is valid
  combine(f, {0, 0, 0, 0}, {0.25, 0.25, 0.25, 0.25}, out.data(),
          scratch.data());
  EXPECT_TRUE(std::isnan(out[1]));
  // A zero weight corner is not read.
  combine(f, {0, 1, 0, 1}, {0.0, 1.0, 0.0, 0.0}, out.data(), scratch.data());
  EXPECT_DOUBLE_EQ(out[0], 3.0);
}

TEST(ReducedGaussianZone, RangesAreSortedAndMerged) {
  Ranges ranges(3, 2);
  ranges << 10, 20, 0, 5, 5, 8;
  const Zone zone(40, ranges);
  EXPECT_EQ(zone.size(), 18);
  EXPECT_EQ(zone.num_ranges(), 2);
  EXPECT_EQ(zone.row(0), 0);
  EXPECT_EQ(zone.row(7), 7);
  EXPECT_EQ(zone.row(8), -1);
  EXPECT_EQ(zone.row(10), 8);
  EXPECT_EQ(zone.row(19), 17);
  EXPECT_EQ(zone.row(20), -1);
  Ranges outside(1, 2);
  outside << 0, 41;
  EXPECT_THROW(Zone(40, outside), std::invalid_argument);
  outside << 5, 5;
  EXPECT_THROW(Zone(40, outside), std::invalid_argument);
}

TEST(ReducedGaussianZone, ABoxAcrossGreenwichHoldsWhatItsPointsRead) {
  const auto mesh = make_n2_mesh();
  const auto zone = Zone::from_box(*mesh, -30.0, -30.0, 30.0, 30.0);
  Cell cell;
  for (int ix = -60; ix <= 60; ++ix) {
    const double lon = 0.5 * ix;
    for (int jx = -60; jx <= 60; ++jx) {
      const double lat = 0.5 * jx;
      ASSERT_TRUE(mesh->locate(lon, lat, cell));
      for (const auto index : cell.index) {
        EXPECT_GE(zone.row(index), 0) << lon << ", " << lat;
      }
    }
  }
  EXPECT_LT(zone.size(), mesh->size());
}

TEST(ReducedGaussianZone, FromPointsHoldsExactlyTheirCorners) {
  const auto mesh = make_n2_mesh();
  Vector<double> lon(2);
  lon << 350.0, 100.0;
  Vector<double> lat(2);
  lat << 0.0, 40.0;
  const auto zone = Zone::from_points(*mesh, lon, lat, 2);
  EXPECT_EQ(zone.size(), 8);
  const auto values = sample(*mesh, [](double x, double y) { return x + y; });
  std::vector<double> held;
  for (const auto index : zone.indices()) {
    held.push_back(values[static_cast<size_t>(index)]);
  }
  const auto global = single(mesh, values);
  const auto regional = single(mesh, held, std::make_shared<const Zone>(zone));
  for (Eigen::Index ix = 0; ix < lon.size(); ++ix) {
    EXPECT_DOUBLE_EQ(interpolate(regional, lon[ix], lat[ix]),
                     interpolate(global, lon[ix], lat[ix]));
  }
  EXPECT_TRUE(std::isnan(interpolate(regional, 200.0, -40.0)));
}

namespace {

/// A series whose step at `times[k]` holds the constant `bases[k]`.
struct ConstantSeries {
  std::shared_ptr<const Mesh> mesh = make_n2_mesh();
  std::vector<std::vector<double>> values;
  Series series{mesh};

  ConstantSeries(const std::vector<int64_t>& times,
                 const std::vector<double>& bases) {
    values.reserve(times.size());
    for (size_t ix = 0; ix < times.size(); ++ix) {
      values.emplace_back(static_cast<size_t>(mesh->size()), bases[ix]);
      series.append(times[ix],
                    std::make_shared<const Grid>(
                        mesh, nullptr, std::vector{field(values.back())}));
    }
  }

  [[nodiscard]] auto at(int64_t time, BracketRule rule = {}) const -> double {
    return interpolate(series, 10.0, 19.4712, time, rule);
  }
};

constexpr int64_t kHour = 3600;

}  // namespace

TEST(ReducedGaussianSeries, LinearAndNearest) {
  const ConstantSeries steps({0, 6 * kHour}, {0.0, 600.0});
  EXPECT_DOUBLE_EQ(steps.at(0), 0.0);
  EXPECT_DOUBLE_EQ(steps.at(2 * kHour), 200.0);
  EXPECT_DOUBLE_EQ(steps.at(6 * kHour), 600.0);
  EXPECT_TRUE(std::isnan(steps.at(-1)));
  EXPECT_TRUE(std::isnan(steps.at((6 * kHour) + 1)));
  const BracketRule nearest{.nearest = true};
  EXPECT_DOUBLE_EQ(steps.at(2 * kHour, nearest), 0.0);
  EXPECT_DOUBLE_EQ(steps.at(4 * kHour, nearest), 600.0);
  EXPECT_DOUBLE_EQ(steps.at(3 * kHour, nearest), 0.0);  // tie: the earlier
}

TEST(ReducedGaussianSeries, TheWeightResolvesTheUnitOfTheDates) {
  // Nanoseconds since 2025-01-01: the weight is formed from the integer
  // difference, so a nanosecond past a bound is resolved.
  constexpr int64_t kOrigin = 1735689600LL * 1000000000LL;
  constexpr int64_t kSix = 6LL * 3600LL * 1000000000LL;
  const ConstantSeries steps({kOrigin, kOrigin + kSix}, {0.0, 100.0});
  for (const int64_t ns : {1, 10, 100, 1000}) {
    EXPECT_DOUBLE_EQ(steps.at(kOrigin + 2000000000LL + ns),
                     100.0 * static_cast<double>(2000000000LL + ns) /
                         static_cast<double>(kSix));
  }
}

TEST(ReducedGaussianSeries, AForecastSeriesIsReadOnItsOwnIntervals) {
  const ConstantSeries steps({9 * kHour, 15 * kHour}, {900.0, 1500.0});
  const BracketRule rule{.interval = 6 * kHour};
  EXPECT_DOUBLE_EQ(steps.at(12 * kHour, rule), 1200.0);
  EXPECT_DOUBLE_EQ(steps.at(14 * kHour, rule), 1400.0);
}

TEST(ReducedGaussianSeries, AMissingBoundIsServedByTheOther) {
  // The 06:00 step could not be fetched.
  const ConstantSeries steps({0, 12 * kHour}, {0.0, 1200.0});
  const BracketRule rule{.interval = 6 * kHour};
  EXPECT_DOUBLE_EQ(steps.at(9 * kHour, rule), 1200.0);   // [06, 12)
  EXPECT_DOUBLE_EQ(steps.at(6 * kHour, rule), 1200.0);   // on the hole
  EXPECT_DOUBLE_EQ(steps.at(3 * kHour, rule), 0.0);      // [00, 06)
  EXPECT_DOUBLE_EQ(steps.at(12 * kHour, rule), 1200.0);  // on a step
  // Without the lattice, the two steps that bracket the date are blended.
  EXPECT_DOUBLE_EQ(steps.at(9 * kHour), 900.0);
}

TEST(ReducedGaussianSeries, AnIntervalWithNoBoundYieldsNothing) {
  const ConstantSeries steps({0, 6 * kHour}, {0.0, 600.0});
  const BracketRule rule{.interval = 6 * kHour};
  EXPECT_TRUE(std::isnan(steps.at(15 * kHour, rule)));  // [12, 18)
  EXPECT_TRUE(std::isnan(steps.at(-9 * kHour, rule)));  // [-12, -6)
  // [-6, 00) kept its later bound, which serves the date alone.
  EXPECT_DOUBLE_EQ(steps.at(-3 * kHour, rule), 0.0);
}

TEST(ReducedGaussianSeries, AWidePairIsAMissingStep) {
  const ConstantSeries steps({0, 6 * kHour, 18 * kHour}, {0.0, 6.0, 18.0});
  const BracketRule rule{.max_step_gap = 6 * kHour};
  EXPECT_DOUBLE_EQ(steps.at(3 * kHour, rule), 3.0);
  EXPECT_TRUE(std::isnan(steps.at(9 * kHour, rule)));
  EXPECT_DOUBLE_EQ(steps.at(18 * kHour, rule), 18.0);
}

TEST(ReducedGaussianSeries, AppendValidatesAndSorts) {
  ConstantSeries steps({6 * kHour, 0}, {6.0, 0.0});
  EXPECT_EQ(steps.series.times(), (std::vector<int64_t>{0, 6 * kHour}));
  const std::vector<double> values(static_cast<size_t>(steps.mesh->size()));
  const auto grid = std::make_shared<const Grid>(steps.mesh, nullptr,
                                                 std::vector{field(values)});
  EXPECT_THROW(steps.series.append(0, grid), std::invalid_argument);
  auto other_field = field(values);
  other_field.name = "g";
  EXPECT_THROW(steps.series.append(
                   kHour, std::make_shared<const Grid>(
                              steps.mesh, nullptr, std::vector{other_field})),
               std::invalid_argument);
  const auto other_mesh = make_rounding_mesh();
  const std::vector<double> other(static_cast<size_t>(other_mesh->size()));
  EXPECT_THROW(steps.series.append(
                   kHour, std::make_shared<const Grid>(
                              other_mesh, nullptr, std::vector{field(other)})),
               std::invalid_argument);
  steps.series.drop_before(kHour);
  EXPECT_EQ(steps.series.times(), (std::vector<int64_t>{6 * kHour}));
}

TEST(ReducedGaussianGrid, Validation) {
  const auto mesh = make_n2_mesh();
  const std::vector<double> few(10);
  EXPECT_THROW(Grid(mesh, nullptr, {field(few)}), std::invalid_argument);
  EXPECT_THROW(Grid(mesh, nullptr, {}), std::invalid_argument);
  const std::vector<double> values(static_cast<size_t>(mesh->size()));
  EXPECT_THROW(Grid(mesh, nullptr, {field(values), field(values)}),
               std::invalid_argument);
  EXPECT_THROW(Grid(mesh, nullptr, {field(values)}, Vector<double>(2),
                    Vector<double>(3)),
               std::invalid_argument);
}

}  // namespace pyinterp::geometry::reduced_gaussian
