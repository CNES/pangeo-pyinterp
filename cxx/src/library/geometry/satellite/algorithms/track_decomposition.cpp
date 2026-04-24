#include "pyinterp/geometry/satellite/algorithms/track_decomposition.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
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

// ---------------------------------------------------------------------------
// Swath-aware bounding box computation
// ---------------------------------------------------------------------------

/// Compute the bounding box of a track segment, optionally expanded for
/// swath width.
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
  auto const sub_lon = lon.subspan(first, last - first + 1);
  auto const sub_lat = lat.subspan(first, last - first + 1);

  auto const [lon_min_it, lon_max_it] = std::ranges::minmax_element(sub_lon);
  auto const [lat_min_it, lat_max_it] = std::ranges::minmax_element(sub_lat);

  auto bbox = BoundingBox{
      geometry::geographic::Point{*lon_min_it, *lat_min_it},
      geometry::geographic::Point{*lon_max_it, *lat_max_it},
  };

  if (swath_width_km > 0.0) {
    auto const half_width_km = swath_width_km / 2.0;

    // Latitude expansion: half-width projected along the meridian.
    auto const dlat = math::degrees(half_width_km / kEarthRadiusKm);
    bbox.min_corner().lat() -= dlat;
    bbox.max_corner().lat() += dlat;

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
    auto const dlon_south = dlon_at(bbox.min_corner().lat());
    auto const dlon_north = dlon_at(bbox.max_corner().lat());
    auto const dlon_max = std::max(dlon_south, dlon_north);

    bbox.min_corner().lon() -= dlon_max;
    bbox.max_corner().lon() += dlon_max;
  }

  return bbox;
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

/// Step 2 of the spec: segment by contiguous latitude-zone class.
auto segment_by_latitude_bands(const Eigen::Ref<const Eigen::VectorXd>& lat,
                               double south_limit, double north_limit)
    -> std::vector<RawBlock> {
  auto const n = static_cast<size_t>(lat.size());
  assert(n > 0);

  std::vector<RawBlock> raw;
  auto cur = RawBlock{
      .first1 = 0,
      .last1 = 0,
      .zone = classify_latitude(lat[0], south_limit, north_limit),
  };

  for (size_t i = 1; i < n; ++i) {
    auto const zone = classify_latitude(lat[i], south_limit, north_limit);
    if (zone != cur.zone) {
      cur.last1 = i - 1;
      raw.push_back(cur);
      cur = RawBlock{.first1 = i, .last1 = i, .zone = zone};
    }
  }
  cur.last1 = n - 1;
  raw.push_back(cur);
  return raw;
}

/// Segment by latitude monotonicity: split wherever latitude direction
/// changes (ascending ↔ descending).
auto segment_by_monotonicity(std::span<const double> lat)
    -> std::vector<RawBlock> {
  auto const n = lat.size();
  assert(n > 0);

  if (n <= 2) {
    // A segment of 1-2 points is trivially monotonic.
    return {{.first1 = 0, .last1 = n - 1, .zone = LatitudeZone::kMid}};
  }

  std::vector<RawBlock> raw;
  size_t seg_start = 0;

  // Find initial direction (skip plateaus).
  enum class Dir : uint8_t { kNone, kAsc, kDesc };
  auto direction = Dir::kNone;

  for (size_t i = 1; i < n; ++i) {
    auto const diff = lat[i] - lat[i - 1];

    auto new_dir = Dir::kNone;
    if (diff > 0.0) {
      new_dir = Dir::kAsc;
    } else if (diff < 0.0) {
      new_dir = Dir::kDesc;
    }

    if (new_dir != Dir::kNone) {
      if (direction != Dir::kNone && new_dir != direction) {
        // Direction reversal: close segment at previous point.
        raw.push_back(RawBlock{
            .first1 = seg_start,
            .last1 = i - 1,
            .zone = LatitudeZone::kMid,  // will be recomputed later
        });
        seg_start = i - 1;  // overlap by one point for continuity
      }
      direction = new_dir;
    }
  }

  // Close final segment.
  raw.push_back(RawBlock{
      .first1 = seg_start,
      .last1 = n - 1,
      .zone = LatitudeZone::kMid,
  });

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
[[nodiscard]] constexpr auto merged_with(BoundingBox const& lhs,
                                         BoundingBox const& rhs) noexcept
    -> BoundingBox {
  return {
      {std::min(rhs.min_corner().lon(), lhs.min_corner().lon()),
       std::min(rhs.min_corner().lat(), lhs.min_corner().lat())},
      {std::max(rhs.max_corner().lon(), lhs.max_corner().lon()),
       std::max(rhs.max_corner().lat(), lhs.max_corner().lat())},
  };
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
    auto const d = lon[i + 1] - lon[i];
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
  std::vector<RawBlock> raw;
  switch (strategy) {
    case DecompositionStrategy::kLatitudeBands:
      raw = segment_by_latitude_bands(lat, options.south_limit(),
                                      options.north_limit());
      break;
    case DecompositionStrategy::kMonotonicSegments:
      raw = segment_by_monotonicity(lat);
      break;
  }

  // Step 2: Apply edge-merge policy.
  raw = apply_edge_merge(std::move(raw), options.min_edge_size());

  // Step 3: Enrich into TrackSegments with bounding boxes.
  std::vector<TrackSegment> segments;
  segments.reserve(raw.size());

  for (auto const& r : raw) {
    auto bbox =
        compute_bbox(lon, lat, r.first1, r.last1, options.swath_width_km());

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
    auto const lon_overlap =
        seg.bbox.min_corner().lon() <= extent.max_corner().lon() &&
        seg.bbox.max_corner().lon() >= extent.min_corner().lon();
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
