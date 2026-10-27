#include "pyinterp/geometry/satellite/algorithms/track_decomposition.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <span>

#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/math.hpp"

namespace pyinterp::geometry::satellite::algorithms {
namespace {

/// Mean Earth radius in kilometers (WGS-84 volumetric mean).
inline constexpr double kEarthRadiusKm = 6371.0;

// ---------------------------------------------------------------------------
// Internal raw block representation (pre-enrichment)
// ---------------------------------------------------------------------------

struct RawBlock {
  size_t first1{};
  size_t last1{};
  LatitudeZone zone{};

  [[nodiscard]] constexpr auto size() const noexcept -> size_t {
    return last1 - first1 + 1;
  }
};

/// Degrees in a full turn of longitude.
inline constexpr double kCircle = 360.0;

/// Half a turn of longitude, in degrees.
inline constexpr double kHalfCircle = 180.0;

/// The shortest arc of longitude a difference stands for, in [-180°, 180°):
/// the ±360° step an antimeridian crossing puts in a raw difference is not
/// travel in the opposite direction.
[[nodiscard]] auto shortest_arc(double delta) noexcept -> double {
  return delta - (kCircle * std::floor((delta + kHalfCircle) / kCircle));
}

/// The box of a box whose longitudes may lie outside [-180°, 180°): its
/// western edge is brought into [-180°, 180°) and its eastern one follows,
/// so the box keeps the arc it covers. A box going round the globe becomes
/// the whole circle.
[[nodiscard]] auto make_box(double west, double south, double east,
                            double north) -> BoundingBox {
  if (east - west >= kCircle) {
    return {{-kHalfCircle, south}, {kHalfCircle, north}};
  }
  auto const shift = west - shortest_arc(west);
  return {{west - shift, south}, {east - shift, north}};
}

/// True if no measurement of a segment had a position: its box is NaN.
[[nodiscard]] auto is_unlocated(BoundingBox const& box) noexcept -> bool {
  return std::isnan(box.min_corner().lon());
}

// ---------------------------------------------------------------------------
// Swath-aware bounding box computation
// ---------------------------------------------------------------------------

/// Compute the bounding box of a track segment, optionally expanded for
/// swath width.
///
/// The longitudes are followed along the segment, each measurement placed by
/// its shortest arc from the one before, so a segment crossing the
/// antimeridian (or Greenwich, for longitudes in [0°, 360°)) gets the arc it
/// occupies and not its complement. The western edge lies in [-180°, 180°)
/// and the eastern one may exceed 180°. A measurement without a position — a
/// NaN longitude or latitude — is left out of the box; a segment without any
/// gets a NaN box.
///
/// For swath expansion, we compute the longitude offset at each edge's
/// latitude independently.  At latitude φ, a cross-track distance d (km)
/// translates to:  Δlon = d / (R·cos(φ))  radians.
///
/// This gives tighter boxes than using the worst-case latitude for the
/// entire segment, at essentially zero extra cost.
auto compute_bbox(std::span<const double> lon, std::span<const double> lat,
                  size_t first, size_t last, double swath_width_km)
    -> BoundingBox {
  bool located = false;
  double previous = 0.0;
  double unwrapped = 0.0;
  double west = 0.0;
  double east = 0.0;
  double south = 0.0;
  double north = 0.0;
  for (auto ix = first; ix <= last; ++ix) {
    if (!std::isfinite(lon[ix]) || !std::isfinite(lat[ix])) {
      continue;
    }
    if (!located) {
      located = true;
      unwrapped = lon[ix];
      west = east = unwrapped;
      south = north = lat[ix];
    } else {
      unwrapped += shortest_arc(lon[ix] - previous);
      west = std::min(west, unwrapped);
      east = std::max(east, unwrapped);
      south = std::min(south, lat[ix]);
      north = std::max(north, lat[ix]);
    }
    previous = lon[ix];
  }
  if (!located) {
    constexpr auto kNaN = std::numeric_limits<double>::quiet_NaN();
    return {{kNaN, kNaN}, {kNaN, kNaN}};
  }

  if (swath_width_km > 0.0) {
    auto const half_width_km = swath_width_km / 2.0;

    // Latitude expansion: half-width projected along the meridian.
    auto const dlat = math::degrees(half_width_km / kEarthRadiusKm);
    south -= dlat;
    north += dlat;

    // Longitude expansion at each latitude edge.
    // Guard against cos(φ) → 0 near poles: clamp to a minimum cos value
    // corresponding to ~89.5° latitude.
    constexpr double kMinCos = 0.0087;  // cos(89.5°)

    auto dlon_at = [&](double lat_deg) -> double {
      auto const cos_lat =
          std::max(std::abs(std::cos(math::radians(lat_deg))), kMinCos);
      return math::degrees(half_width_km / (kEarthRadiusKm * cos_lat));
    };

    // Expand using the latitude that produces the widest offset on each side.
    // For the southern edge, the latitude closer to a pole gives wider offset.
    // For the northern edge, same logic.  Since we want the outermost
    // expansion, we pick the larger of the two edge latitudes' expansions
    // for each side.
    auto const dlon_max = std::max(dlon_at(south), dlon_at(north));
    west -= dlon_max;
    east += dlon_max;
  }

  return make_box(west, south, east, north);
}

/// Determine the dominant latitude zone for a segment by majority vote.
auto dominant_zone(const Eigen::Ref<const Eigen::VectorXd>& lat, size_t first,
                   size_t last, double south_limit, double north_limit)
    -> LatitudeZone {
  auto counts = std::array<size_t, 3>{};
  for (auto i = first; i <= last; ++i) {
    auto const z = classify_latitude(lat[i], south_limit, north_limit);
    ++counts[static_cast<size_t>(z)];
  }
  auto const max_it = std::ranges::max_element(counts);
  return static_cast<LatitudeZone>(std::distance(std::begin(counts), max_it));
}

// ---------------------------------------------------------------------------
// Raw block segmentation
// ---------------------------------------------------------------------------

/// Cut the track into raw blocks wherever the latitude turns back and, when
/// @p by_zone is set, wherever the latitude zone changes.
///
/// With @p by_zone set, the latitude is only watched turning back inside the
/// MID zone: between the thresholds a pass goes one way, so a latitude coming
/// back there is the next pass, met without the polar one that separates them
/// — a gap in the data. The two passes lie half a turn of longitude apart and
/// are not one block. Without it, every turn of the latitude opens a block.
///
/// The sample the latitude turns back on belongs to either side by its
/// latitude alone: it goes with the neighbour nearer in longitude. Blocks never
/// overlap. A measurement without a latitude tells nothing of the trend.
auto segment_track(std::span<const double> lon, std::span<const double> lat,
                   bool by_zone, double south_limit, double north_limit)
    -> std::vector<RawBlock> {
  auto const n = lat.size();
  assert(n > 0);

  auto const classify = [&](double value) -> LatitudeZone {
    return by_zone ? classify_latitude(value, south_limit, north_limit)
                   : LatitudeZone::kMid;
  };
  // The arc of longitude between two samples, in degrees.
  auto const arc = [&lon](size_t from, size_t to) -> double {
    return std::abs(shortest_arc(lon[to] - lon[from]));
  };

  std::vector<RawBlock> raw;
  auto zone = classify(lat[0]);
  size_t seg_start = 0;
  // The last latitude of the current block, and the way its latitudes go:
  // +1 northwards, -1 southwards, 0 as long as nothing tells.
  auto previous = lat[0];
  auto trend = 0;

  for (size_t i = 1; i < n; ++i) {
    auto const value = lat[i];
    auto const classified = classify(value);
    auto const step =
        static_cast<int>(value > previous) - static_cast<int>(value < previous);
    auto const turned_back =
        zone == LatitudeZone::kMid && step != 0 && step == -trend;
    if (classified != zone || turned_back) {
      auto first = i;
      if (turned_back && i - 1 > seg_start &&
          arc(i - 2, i - 1) > arc(i - 1, i)) {
        first = i - 1;
      }
      raw.push_back(
          RawBlock{.first1 = seg_start, .last1 = first - 1, .zone = zone});
      seg_start = first;
      zone = classified;
      trend = first == i ? 0 : step;
    } else if (step != 0) {
      trend = step;
    }
    if (!std::isnan(value)) {
      previous = value;
    }
  }
  raw.push_back(RawBlock{.first1 = seg_start, .last1 = n - 1, .zone = zone});
  return raw;
}

// ---------------------------------------------------------------------------
// Edge-merge policy
// ---------------------------------------------------------------------------

/// Merge two raw blocks into one. Zone is inherited from the larger block.
auto merge_raw(RawBlock const& a, RawBlock const& b) -> RawBlock {
  auto const zone = (a.size() >= b.size()) ? a.zone : b.zone;
  return {
      .first1 = a.first1,
      .last1 = b.last1,
      .zone = zone,
  };
}

/// Apply the edge-size merge policy from the spec.
auto apply_edge_merge(std::vector<RawBlock> raw, size_t min_edge_size)
    -> std::vector<RawBlock> {
  auto const k = raw.size();
  if (k <= 1) {
    return raw;
  }

  auto const small_first = raw.front().size() < min_edge_size;
  auto const small_last = raw.back().size() < min_edge_size;

  if (k == 2) {
    if (small_first || small_last) {
      return {merge_raw(raw[0], raw[1])};
    }
    return raw;
  }

  if (k == 3) {
    if (small_first && small_last) {
      return {merge_raw(merge_raw(raw[0], raw[1]), raw[2])};
    }
    if (small_first) {
      return {merge_raw(raw[0], raw[1]), raw[2]};
    }
    if (small_last) {
      return {raw[0], merge_raw(raw[1], raw[2])};
    }
    return raw;
  }

  // k >= 4
  std::vector<RawBlock> result;
  size_t start = 0;

  if (small_first) {
    result.push_back(merge_raw(raw[0], raw[1]));
    start = 2;
  }

  auto const merge_end = small_last ? k - 2 : k;
  for (size_t i = start; i < merge_end; ++i) {
    result.push_back(raw[i]);
  }

  if (small_last) {
    result.push_back(merge_raw(raw[k - 2], raw[k - 1]));
  }

  return result;
}

// Geodetic area estimate on the sphere (km²).
// Uses the spherical cap formula: A = R² |sin(φ2) - sin(φ1)| |Δλ|.
[[nodiscard]] __CONSTEXPR auto area_km2(const BoundingBox& box) noexcept
    -> double {
  auto const dlam =
      math::radians(box.max_corner().lon() - box.min_corner().lon());
  auto const dsin = std::abs(std::sin(math::radians(box.max_corner().lat())) -
                             std::sin(math::radians(box.min_corner().lat())));
  return kEarthRadiusKm * kEarthRadiusKm * std::abs(dlam) * dsin;
}

// Merge two bounding boxes into the smallest enclosing box.
//
// The longitudes of a box follow its track and may exceed 180°; the second
// box is brought beside the first before the two are joined, so that two boxes
// either side of the antimeridian give the arc between them and not its
// complement. A box without a position is ignored.
[[nodiscard]] auto merged_with(BoundingBox const& lhs,
                               BoundingBox const& rhs) noexcept -> BoundingBox {
  if (is_unlocated(lhs)) {
    return rhs;
  }
  if (is_unlocated(rhs)) {
    return lhs;
  }
  // The whole turns that bring the centre of the second beside the first's.
  auto const lhs_centre =
      0.5 * (lhs.min_corner().lon() + lhs.max_corner().lon());
  auto const rhs_centre =
      0.5 * (rhs.min_corner().lon() + rhs.max_corner().lon());
  auto const turn = kCircle * std::round((lhs_centre - rhs_centre) / kCircle);
  return make_box(
      std::min(lhs.min_corner().lon(), rhs.min_corner().lon() + turn),
      std::min(lhs.min_corner().lat(), rhs.min_corner().lat()),
      std::max(lhs.max_corner().lon(), rhs.max_corner().lon() + turn),
      std::max(lhs.max_corner().lat(), rhs.max_corner().lat()));
}

// True if the arcs of longitude [a_min, a_max] and [b_min, b_max] share a
// meridian, whatever turn of the circle each is written on.
[[nodiscard]] auto longitudes_overlap(double a_min, double a_max, double b_min,
                                      double b_max) noexcept -> bool {
  if (a_max - a_min >= kCircle || b_max - b_min >= kCircle) {
    return true;
  }
  // The western edge of a, on the turn that starts at the western edge of b.
  auto const west = a_min - (kCircle * std::floor((a_min - b_min) / kCircle));
  return west <= b_max || west + (a_max - a_min) >= b_min + kCircle;
}

// ---------------------------------------------------------------------------
// Area-based merge pass (for monotonic strategy)
// ---------------------------------------------------------------------------

auto apply_area_merge(std::vector<TrackSegment> segments,
                      double merge_area_ratio, size_t max_segments)
    -> std::vector<TrackSegment> {
  if (segments.size() <= 1) {
    return segments;
  }

  // A ratio of 0 disables area-based merging entirely.
  // Still honor max_segments if set.
  if (merge_area_ratio <= 0.0 &&
      (max_segments == 0 || segments.size() <= max_segments)) {
    return segments;
  }

  // Iteratively merge adjacent pairs when doing so doesn't blow up the
  // total bounding-box area.  The `merge_area_ratio` controls how much
  // area increase is acceptable: a pair is merged when
  //   area(merged_bbox) / (area(A) + area(B))  <=  merge_area_ratio
  //
  // A ratio of ~1.0 means the merged box is about the same size as the
  // two individual boxes combined (significant overlap).  A ratio of 1.5
  // allows 50% area increase.  Setting it very high (e.g. 999) effectively
  // merges everything — which is usually NOT what you want.
  //
  // If `max_segments > 0` and we still have too many segments after the
  // ratio-based pass, we force-merge the least-costly pair.

  bool progress = true;
  while (progress && segments.size() > 1) {
    progress = false;

    // Find the pair with the lowest merge ratio.
    double best_ratio = std::numeric_limits<double>::infinity();
    size_t best_idx = 0;

    for (size_t i = 0; i + 1 < segments.size(); ++i) {
      auto const& a = segments[i];
      auto const& b = segments[i + 1];
      auto const sum_area = area_km2(a.bbox) + area_km2(b.bbox);
      if (sum_area <= 0.0) {
        continue;
      }
      auto const merged_area = area_km2(merged_with(a.bbox, b.bbox));
      auto const ratio = merged_area / sum_area;
      if (ratio < best_ratio) {
        best_ratio = ratio;
        best_idx = i;
      }
    }

    // Merge if the best pair is within the allowed ratio.
    bool do_merge = (best_ratio <= merge_area_ratio);

    // If not merging by ratio, but we exceed max_segments, force it.
    if (!do_merge && max_segments > 0 && segments.size() > max_segments) {
      do_merge = true;
      // best_idx already points to the least-costly pair.
    }

    if (do_merge) {
      auto& a = segments[best_idx];
      auto const& b = segments[best_idx + 1];
      a.last_index = b.last_index;
      a.bbox = merged_with(a.bbox, b.bbox);
      if (b.size() > a.size()) {
        a.zone = b.zone;
      }
      segments.erase(segments.begin() +
                     static_cast<std::ptrdiff_t>(best_idx + 1));
      progress = true;
    }
  }

  return segments;
}

}  // namespace

auto infer_orbit_direction(const Eigen::Ref<const Eigen::VectorXd>& lon)
    -> OrbitDirection {
  if (lon.size() < 2) {
    return OrbitDirection::kRetrograde;
  }

  auto const n = std::min(static_cast<size_t>(lon.size()) - 1, kOrbitDiffSize);
  size_t positive = 0;
  size_t negative = 0;

  for (size_t i = 0; i < n; ++i) {
    auto const d = shortest_arc(lon[i + 1] - lon[i]);
    if (d >= 0.0) {
      ++positive;
    }
    if (d <= 0.0) {
      ++negative;
    }
  }

  return (positive > negative) ? OrbitDirection::kPrograde
                               : OrbitDirection::kRetrograde;
}

auto decompose_track(const Eigen::Ref<const Eigen::VectorXd>& lon,
                     const Eigen::Ref<const Eigen::VectorXd>& lat,
                     DecompositionStrategy strategy,
                     DecompositionOptions const& options)
    -> std::vector<TrackSegment> {
  if (lon.size() == 0 || lat.size() == 0) {
    throw std::invalid_argument("decompose_track: lon/lat must be non-empty");
  }
  if (lon.size() != lat.size()) {
    throw std::invalid_argument(
        "decompose_track: lon and lat must have the same size");
  }
  if (options.south_limit() >= options.north_limit()) {
    throw std::invalid_argument(
        "decompose_track: south_limit must be less than north_limit");
  }

  auto const orbit = infer_orbit_direction(lon);

  // Step 1: Segment into raw blocks.
  auto const lon_span = std::span<const double>(lon.data(), lon.size());
  auto const lat_span = std::span<const double>(lat.data(), lat.size());
  auto raw = segment_track(lon_span, lat_span,
                           strategy == DecompositionStrategy::kLatitudeBands,
                           options.south_limit(), options.north_limit());

  // Step 2: Apply edge-merge policy.
  raw = apply_edge_merge(std::move(raw), options.min_edge_size());

  // Step 3: Enrich into TrackSegments with bounding boxes.
  std::vector<TrackSegment> segments;
  segments.reserve(raw.size());

  for (auto const& r : raw) {
    auto bbox = compute_bbox(lon_span, lat_span, r.first1, r.last1,
                             options.swath_width_km());

    // For latitude-band strategy, zone comes from segmentation.
    // For monotonic strategy, recompute dominant zone.
    auto zone = r.zone;
    if (strategy == DecompositionStrategy::kMonotonicSegments) {
      zone = dominant_zone(lat, r.first1, r.last1, options.south_limit(),
                           options.north_limit());
    }

    segments.push_back(TrackSegment{
        .first_index = r.first1,
        .last_index = r.last1,
        .bbox = bbox,
        .zone = zone,
        .orbit = orbit,
    });
  }

  // Step 4: For monotonic strategy, apply area-based merge.
  if (strategy == DecompositionStrategy::kMonotonicSegments) {
    segments = apply_area_merge(std::move(segments), options.merge_area_ratio(),
                                options.max_segments());
  }

  return segments;
}

auto filter_by_extent(std::vector<TrackSegment> const& segments,
                      BoundingBox const& extent) -> std::vector<TrackSegment> {
  std::vector<TrackSegment> result;
  for (auto const& seg : segments) {
    // Two axis-aligned boxes intersect iff they overlap on both axes.
    auto const lon_overlap = longitudes_overlap(
        seg.bbox.min_corner().lon(), seg.bbox.max_corner().lon(),
        extent.min_corner().lon(), extent.max_corner().lon());
    auto const lat_overlap =
        seg.bbox.min_corner().lat() <= extent.max_corner().lat() &&
        seg.bbox.max_corner().lat() >= extent.min_corner().lat();
    if (lon_overlap && lat_overlap) {
      result.push_back(seg);
    }
  }
  return result;
}

}  // namespace pyinterp::geometry::satellite::algorithms
