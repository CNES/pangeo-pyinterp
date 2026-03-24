/// @file track_decomposition.hpp
/// @brief Decompose a satellite ground track (or swath) into segments with
///        tight bounding boxes for efficient grid loading.
///
/// The core idea: a satellite half-orbit projected onto a sphere traces a long
/// curve. Loading the full bounding box of that curve from a Cartesian grid
/// would pull too much data into memory. By splitting the track into segments
/// (typically by latitude zone or monotonicity), each segment gets a much
/// tighter bounding box — reducing memory usage proportionally.
///
/// Two geometry modes are supported:
///   - LineString: the nadir ground track (zero swath width).
///   - Swath: the track buffered by the instrument's cross-track half-width,
///            producing latitude-dependent longitude expansion.
#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pyinterp/geometry/geographic/box.hpp"

namespace pyinterp::geometry::satellite::algorithms {

/// Number of first-difference samples used for orbit direction inference.
inline constexpr size_t kOrbitDiffSize = 10;

/// Latitude-band classification of a track point or segment.
enum class LatitudeZone : uint8_t {
  kSouth,  ///< lat <= south_limit
  kMid,    ///< south_limit < lat < north_limit
  kNorth,  ///< lat >= north_limit
};

/// Orbital direction inferred from longitude progression.
enum class OrbitDirection : uint8_t {
  kPrograde,    ///< Predominantly eastward (ascending node).
  kRetrograde,  ///< Predominantly westward (descending node). Also tie case.
};

/// Strategy for splitting a track into segments.
enum class DecompositionStrategy : uint8_t {
  /// Fixed latitude thresholds producing up to 3 zones (SOUTH/MID/NORTH).
  /// Backward-compatible with the original create_processing_blocks behavior.
  kLatitudeBands,

  /// Split at every latitude direction change (ascending ↔ descending).
  /// Produces naturally tight bounding boxes for near-polar orbits.
  kMonotonicSegments,
};

/// Axis-aligned bounding box in geographic coordinates (degrees).
using BoundingBox = geometry::geographic::Box;

/// A contiguous segment of the original track, with its bounding box.
struct TrackSegment {
  size_t first_index{};    ///< First index in the input arrays (inclusive).
  size_t last_index{};     ///< Last index in the input arrays (inclusive).
  BoundingBox bbox{};      ///< Bounding box (swath-expanded if applicable).
  LatitudeZone zone{};     ///< Dominant latitude zone of this segment.
  OrbitDirection orbit{};  ///< Orbit direction (shared across all segments).

  /// Number of points in this segment.
  [[nodiscard]] constexpr auto size() const noexcept -> size_t {
    return last_index - first_index + 1;
  }
};

/// Parameters controlling track decomposition behavior.
class DecompositionOptions {
 public:
  /// @brief Default constructor
  constexpr DecompositionOptions() = default;

  /// @brief Get the south latitude limit for zone classification.
  /// @return South latitude limit (degrees)
  [[nodiscard]] constexpr auto south_limit() const -> double {
    return south_limit_;
  }

  /// @brief Get the north latitude limit for zone classification.
  /// @return North latitude limit (degrees)
  [[nodiscard]] constexpr auto north_limit() const -> double {
    return north_limit_;
  }

  /// @brief Get the minimum number of points for the first/last segment.
  ///        Segments smaller than this are merged with their neighbor.
  /// @return Minimum edge size (number of points)
  [[nodiscard]] constexpr auto min_edge_size() const -> size_t {
    return min_edge_size_;
  }

  /// @brief Get the cross-track swath full width in kilometers.
  ///        Zero means line-string mode (no lateral expansion). For SWOT, use
  ///        ~140.
  /// @return Swath width (km)
  [[nodiscard]] constexpr auto swath_width_km() const -> double {
    return swath_width_km_;
  }

  /// @brief Get the maximum allowed area ratio for merging segments.
  ///        Adjacent segments are merged when the combined box stays at or
  ///        below this ratio times the sum of individual areas. Set to 0.0 to
  ///        disable area-based merging entirely (only edge-merge policy will
  ///        apply).
  /// @return Merge area ratio
  [[nodiscard]] constexpr auto merge_area_ratio() const -> double {
    return merge_area_ratio_;
  }

  /// @brief Get the maximum number of output segments (0 = unlimited).
  /// @return Maximum number of segments
  [[nodiscard]] constexpr auto max_segments() const -> size_t {
    return max_segments_;
  }

  /// @brief Update the south latitude limit for zone classification.
  /// @param[in] limit New south latitude limit (degrees).
  /// @return Updated `DecompositionOptions` instance.
  [[nodiscard]] constexpr auto with_south_limit(this DecompositionOptions self,
                                                double limit)
      -> DecompositionOptions {
    self.south_limit_ = limit;
    return self;
  }

  /// @brief Update the north latitude limit for zone classification.
  /// @param[in] limit New north latitude limit (degrees).
  /// @return Updated `DecompositionOptions` instance.
  [[nodiscard]] constexpr auto with_north_limit(this DecompositionOptions self,
                                                double limit)
      -> DecompositionOptions {
    self.north_limit_ = limit;
    return self;
  }

  /// @brief Update the minimum number of points for the first/last segment.
  ///        Segments smaller than this are merged with their neighbor.
  /// @param[in] size New minimum edge size (number of points).
  /// @return Updated `DecompositionOptions` instance.
  [[nodiscard]] constexpr auto with_min_edge_size(
      this DecompositionOptions self, size_t size) -> DecompositionOptions {
    self.min_edge_size_ = size;
    return self;
  }

  /// @brief Update the cross-track swath full width in kilometers.
  ///        Zero means line-string mode (no lateral expansion). For SWOT, use
  ///        ~140.
  /// @param[in] width_km New swath width (km).
  /// @return Updated `DecompositionOptions` instance.
  [[nodiscard]] constexpr auto with_swath_width_km(
      this DecompositionOptions self, double width_km) -> DecompositionOptions {
    self.swath_width_km_ = width_km;
    return self;
  }

  /// @brief Update the maximum allowed area ratio for merging segments.
  ///        Adjacent segments are merged when the combined box stays at or
  ///        below this ratio times the sum of individual areas. Set to 0.0 to
  ///        disable area-based merging entirely (only edge-merge policy will
  ///        apply).
  /// @param[in] ratio New merge area ratio.
  /// @return Updated `DecompositionOptions` instance.
  [[nodiscard]] constexpr auto with_merge_area_ratio(
      this DecompositionOptions self, double ratio) -> DecompositionOptions {
    self.merge_area_ratio_ = ratio;
    return self;
  }

  /// @brief Update the maximum number of output segments (0 = unlimited).
  /// @param[in] max_segments New maximum number of segments.
  /// @return Updated `DecompositionOptions` instance.
  [[nodiscard]] constexpr auto with_max_segments(this DecompositionOptions self,
                                                 size_t max_segments)
      -> DecompositionOptions {
    self.max_segments_ = max_segments;
    return self;
  }

 private:
  /// South laitude limit for zone classification (degrees).
  double south_limit_{-50.0};
  /// North latitude limit for zone classification (degrees).
  double north_limit_{50.0};

  /// Minimum number of points for the first/last segment.  Segments smaller
  /// than this are merged with their neighbor.
  size_t min_edge_size_{10};

  /// Cross-track swath full width in kilometers.  Zero means line-string
  /// mode (no lateral expansion).  For SWOT, use ~140.
  double swath_width_km_{0.0};

  /// Maximum allowed ratio: area(merged) / (area(A) + area(B)).
  /// Adjacent segments are merged when the combined box stays at or below
  /// this ratio times the sum of individual areas.
  ///
  /// The ratio is always >= 0.5 (perfect overlap) and approaches 1.0 when
  /// boxes are adjacent with no overlap.  Values > 1.0 allow merges that
  /// increase total area (e.g. 1.5 = accept 50% area increase).
  ///
  /// Set to 0.0 to disable area-based merging entirely (only edge-merge
  /// policy will apply).
  double merge_area_ratio_{1.5};

  /// Maximum number of output segments (0 = unlimited).
  size_t max_segments_{0};
};

/// Classify a latitude value into a zone.
[[nodiscard]] constexpr auto classify_latitude(const double lat,
                                               const double south_limit,
                                               const double north_limit)
    -> LatitudeZone {
  if (lat <= south_limit) {
    return LatitudeZone::kSouth;
  }
  if (lat >= north_limit) {
    return LatitudeZone::kNorth;
  }
  return LatitudeZone::kMid;
}

/// Infer the orbit direction from a longitude sequence using a sign-majority
/// of first differences over the first `kOrbitDiffSize` samples.
///
/// Tie resolves to kRetrograde (preserving original behavior).
///
/// @param[in] lon Longitude sequence (degrees).
/// @return Inferred orbit direction
[[nodiscard]] auto infer_orbit_direction(
    const Eigen::Ref<const Eigen::VectorXd>& lon) -> OrbitDirection;

/// Main entry point: decompose a track into segments with tight bounding
/// boxes.
///
/// @param[in] lon        Longitude array (degrees), size N.
/// @param[in] lat        Latitude array (degrees), size N.  Must equal lon
/// size.
/// @param[in] strategy   Decomposition method.
/// @param[in] options    Strategy-specific parameters.
/// @return Ordered, contiguous, non-overlapping segments whose union
///         covers [0, N-1].
///
/// @throws std::invalid_argument if lon/lat are empty or have different sizes.
[[nodiscard]] auto decompose_track(const Eigen::Ref<const Eigen::VectorXd>& lon,
                                   const Eigen::Ref<const Eigen::VectorXd>& lat,
                                   DecompositionStrategy strategy,
                                   DecompositionOptions const& options = {})
    -> std::vector<TrackSegment>;

/// Convenience overload: given pre-computed segments and a data extent,
/// return only the segments whose bounding box intersects the extent.
/// Useful for filtering out segments that fall entirely outside a grid.
/// @param[in] segments Track segments to filter.
/// @param[in] data_extent Bounding box of the data grid.
/// @return Subset of input segments that intersect the data extent.
[[nodiscard]] auto filter_by_extent(std::vector<TrackSegment> const& segments,
                                    BoundingBox const& data_extent)
    -> std::vector<TrackSegment>;

}  // namespace pyinterp::geometry::satellite::algorithms
