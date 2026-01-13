// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <boost/geometry.hpp>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kToGeojsonDoc = R"doc(
Converts a geometry to GeoJSON representation.

GeoJSON is a format for encoding geographic data structures using JSON.
This function converts Boost.Geometry objects to their GeoJSON equivalent.

Args:
    geometry: The geometry to convert.

Returns:
    A JSON string representing the geometry in GeoJSON format.

Examples:
    >>> point = Point(1.0, 2.0)
    >>> geojson = to_geojson(point)
    >>> # Returns '{"type":"Point","coordinates":[1.0,2.0]}'
)doc";

constexpr auto kFromGeojsonDoc = R"doc(
Creates a geometry from GeoJSON representation.

Parses a GeoJSON string and constructs the corresponding geometry object.
The function automatically detects the geometry type from the GeoJSON "type"
field and returns the appropriate geometry object.

Args:
    geojson: GeoJSON string representation of the geometry.

Returns:
    A variant containing one of: Point, LineString, Ring, Polygon, MultiPoint,
    MultiLineString, or MultiPolygon depending on the GeoJSON type field.

Examples:
    >>> geojson_str = '{"type":"Point","coordinates":[1.0,2.0]}'
    >>> point = from_geojson(geojson_str)
    >>> # Returns Point(1.0, 2.0)

    >>> geojson_str = '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}'
    >>> polygon = from_geojson(geojson_str)
    >>> # Returns Polygon object
)doc";

/// @brief Default precision for GeoJSON coordinate output
inline constexpr int kGeoJsonPrecision = 17;

/// @brief Helper to convert a Point to GeoJSON coordinates array
/// @tparam Point Boost.Geometry point type
/// @param[in] pt The point to convert
/// @return JSON array string representing the coordinates
template <typename Point>
[[nodiscard]] inline auto point_to_coords(const Point& pt) -> std::string {
  std::ostringstream oss;
  oss << std::setprecision(kGeoJsonPrecision) << "["
      << boost::geometry::get<0>(pt) << "," << boost::geometry::get<1>(pt)
      << "]";
  return oss.str();
}

/// @brief Helper to convert a range of points to GeoJSON coordinates array
/// @tparam Range Range type containing points
/// @param[in] range The range of points to convert
/// @return JSON array string representing the coordinates
template <typename Range>
[[nodiscard]] inline auto points_range_to_coords(const Range& range)
    -> std::string {
  std::ostringstream oss;
  oss << "[";
  bool first = true;
  for (const auto& pt : range) {
    if (!first) {
      oss << ",";
    }
    oss << point_to_coords(pt);
    first = false;
  }
  oss << "]";
  return oss.str();
}

/// @brief Helper to convert LineString to GeoJSON coordinates array
/// @tparam LineString Boost.Geometry linestring type
/// @param[in] ls The linestring to convert
/// @return JSON array string representing the coordinates
template <typename LineString>
[[nodiscard]] inline auto linestring_to_coords(const LineString& ls)
    -> std::string {
  return points_range_to_coords(ls);
}

/// @brief Helper to convert Ring to GeoJSON coordinates array
/// @tparam Ring Boost.Geometry ring type
/// @param[in] ring The ring to convert
/// @return JSON array string representing the coordinates
template <typename Ring>
[[nodiscard]] inline auto ring_to_coords(const Ring& ring) -> std::string {
  return points_range_to_coords(ring);
}

/// @brief Helper to convert Polygon to GeoJSON coordinates array
/// @tparam Polygon Boost.Geometry polygon type
/// @param[in] poly The polygon to convert
/// @return JSON array string representing the coordinates (outer ring + holes)
template <typename Polygon>
[[nodiscard]] inline auto polygon_to_coords(const Polygon& poly)
    -> std::string {
  std::ostringstream oss;
  oss << "[";
  // Outer ring
  oss << ring_to_coords(poly.outer());
  // Inner rings (holes)
  for (const auto& inner : poly.inners()) {
    oss << "," << ring_to_coords(inner);
  }
  oss << "]";
  return oss.str();
}

/// @brief Convert Point to GeoJSON
/// @tparam Point Boost.Geometry point type
/// @param[in] pt The point to convert
/// @return GeoJSON string representation
template <typename Point>
[[nodiscard]] inline auto point_to_geojson(const Point& pt) -> std::string {
  std::ostringstream oss;
  oss << R"({"type":"Point","coordinates":)" << point_to_coords(pt) << "}";
  return oss.str();
}

/// @brief Convert LineString to GeoJSON
/// @tparam LineString Boost.Geometry linestring type
/// @param[in] ls The linestring to convert
/// @return GeoJSON string representation
template <typename LineString>
[[nodiscard]] inline auto linestring_to_geojson(const LineString& ls)
    -> std::string {
  std::ostringstream oss;
  oss << R"({"type":"LineString","coordinates":)" << linestring_to_coords(ls)
      << "}";
  return oss.str();
}

/// @brief Convert Ring to GeoJSON (represented as LineString per GeoJSON spec)
/// @tparam Ring Boost.Geometry ring type
/// @param[in] ring The ring to convert
/// @return GeoJSON string representation as LineString type
template <typename Ring>
[[nodiscard]] inline auto ring_to_geojson(const Ring& ring) -> std::string {
  std::ostringstream oss;
  oss << R"({"type":"LineString","coordinates":)" << ring_to_coords(ring)
      << "}";
  return oss.str();
}

/// @brief Convert Polygon to GeoJSON
/// @tparam Polygon Boost.Geometry polygon type
/// @param[in] poly The polygon to convert
/// @return GeoJSON string representation
template <typename Polygon>
[[nodiscard]] inline auto polygon_to_geojson(const Polygon& poly)
    -> std::string {
  std::ostringstream oss;
  oss << R"({"type":"Polygon","coordinates":)" << polygon_to_coords(poly)
      << "}";
  return oss.str();
}

/// @brief Convert MultiPoint to GeoJSON
/// @tparam MultiPoint Boost.Geometry multi_point type
/// @param[in] mp The multi-point to convert
/// @return GeoJSON string representation
template <typename MultiPoint>
[[nodiscard]] inline auto multipoint_to_geojson(const MultiPoint& mp)
    -> std::string {
  std::ostringstream oss;
  oss << R"({"type":"MultiPoint","coordinates":)" << points_range_to_coords(mp)
      << "}";
  return oss.str();
}

/// @brief Convert MultiLineString to GeoJSON
/// @tparam MultiLineString Boost.Geometry multi_linestring type
/// @param[in] mls The multi-linestring to convert
/// @return GeoJSON string representation
template <typename MultiLineString>
[[nodiscard]] inline auto multilinestring_to_geojson(const MultiLineString& mls)
    -> std::string {
  std::ostringstream oss;
  oss << R"({"type":"MultiLineString","coordinates":[)";
  bool first = true;
  for (const auto& ls : mls) {
    if (!first) {
      oss << ",";
    }
    oss << linestring_to_coords(ls);
    first = false;
  }
  oss << "]}";
  return oss.str();
}

/// @brief Convert MultiPolygon to GeoJSON
/// @tparam MultiPolygon Boost.Geometry multi_polygon type
/// @param[in] mpoly The multi-polygon to convert
/// @return GeoJSON string representation
template <typename MultiPolygon>
[[nodiscard]] inline auto multipolygon_to_geojson(const MultiPolygon& mpoly)
    -> std::string {
  std::ostringstream oss;
  oss << R"({"type":"MultiPolygon","coordinates":[)";
  bool first = true;
  for (const auto& poly : mpoly) {
    if (!first) {
      oss << ",";
    }
    oss << polygon_to_coords(poly);
    first = false;
  }
  oss << "]}";
  return oss.str();
}

/// @brief Macro for all geometry types supporting GeoJSON
#define GEOJSON_TYPES(NS)                                           \
  NS::Point, NS::LineString, NS::Ring, NS::Polygon, NS::MultiPoint, \
      NS::MultiLineString, NS::MultiPolygon

/// @brief Helper to define to_geojson for geometry types
/// @tparam Geometries Geometry types
/// @param[in] m Python module
/// @param[in] doc Documentation string
template <typename... Geometries>
inline auto define_to_geojson(nanobind::module_& m, const char* doc) -> void {
  (...,
   m.def(
       "to_geojson",
       [](const Geometries& g) -> std::string {
         nanobind::gil_scoped_release release;
         // Dispatch based on geometry type tag
         using Tag = typename boost::geometry::traits::tag<Geometries>::type;
         if constexpr (std::is_same_v<Tag, boost::geometry::point_tag>) {
           return point_to_geojson(g);
         } else if constexpr (std::is_same_v<Tag,
                                             boost::geometry::linestring_tag>) {
           return linestring_to_geojson(g);
         } else if constexpr (std::is_same_v<Tag, boost::geometry::ring_tag>) {
           return ring_to_geojson(g);
         } else if constexpr (std::is_same_v<Tag,
                                             boost::geometry::polygon_tag>) {
           return polygon_to_geojson(g);
         } else if constexpr (std::is_same_v<
                                  Tag, boost::geometry::multi_point_tag>) {
           return multipoint_to_geojson(g);
         } else if constexpr (std::is_same_v<
                                  Tag, boost::geometry::multi_linestring_tag>) {
           return multilinestring_to_geojson(g);
         } else if constexpr (std::is_same_v<
                                  Tag, boost::geometry::multi_polygon_tag>) {
           return multipolygon_to_geojson(g);
         }
         return "{}";
       },
       "geometry"_a, doc));
}

/// @brief Simple JSON parser helpers for GeoJSON
/// @details Provides minimal JSON parsing sufficient for GeoJSON geometry
/// coordinates. This is not a full JSON parser and only handles the subset
/// needed for GeoJSON geometry objects.
namespace json {

/// @brief Maximum nesting depth for JSON structures to prevent stack overflow
inline constexpr size_t kMaxNestingDepth = 100;

/// @brief Skip whitespace characters in JSON string
/// @param[in] str The JSON string
/// @param[in,out] pos Current position (updated to skip whitespace)
inline auto skip_whitespace(std::string_view str, size_t& pos) noexcept
    -> void {
  while (pos < str.size()) {
    const char c = str[pos];
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
      break;
    }
    ++pos;
  }
}

/// @brief Parse a JSON number value
/// @param[in] str The JSON string
/// @param[in,out] pos Current position (updated past the number)
/// @return The parsed double value
/// @throws std::runtime_error if no valid number found
[[nodiscard]] inline auto parse_number(std::string_view str, size_t& pos)
    -> double {
  if (pos >= str.size()) {
    throw std::runtime_error("Unexpected end of input while parsing number");
  }
  const size_t start = pos;
  // Handle optional leading minus
  if (pos < str.size() && str[pos] == '-') {
    ++pos;
  }
  // Parse digits, decimal point, and exponent
  while (pos < str.size()) {
    const char c = str[pos];
    if (!std::isdigit(static_cast<unsigned char>(c)) && c != '.' && c != 'e' &&
        c != 'E' && c != '+' && c != '-') {
      break;
    }
    // Only allow sign after 'e' or 'E'
    if ((c == '+' || c == '-') && pos > start && str[pos - 1] != 'e' &&
        str[pos - 1] != 'E') {
      break;
    }
    ++pos;
  }
  if (pos == start) {
    throw std::runtime_error("Invalid number format in JSON");
  }
  return std::stod(std::string(str.substr(start, pos - start)));
}

/// @brief Parse a JSON string value (without escape sequence processing)
/// @param[in] str The JSON string
/// @param[in,out] pos Current position (updated past the closing quote)
/// @return The parsed string content (without quotes)
/// @throws std::runtime_error if string is malformed
[[nodiscard]] inline auto parse_string(std::string_view str, size_t& pos)
    -> std::string {
  if (pos >= str.size() || str[pos] != '"') {
    throw std::runtime_error("Expected '\"' at start of string at position " +
                             std::to_string(pos));
  }
  ++pos;
  const size_t start = pos;
  while (pos < str.size() && str[pos] != '"') {
    if (str[pos] == '\\') {
      ++pos;  // Skip escaped character
      if (pos >= str.size()) {
        throw std::runtime_error("Unterminated escape sequence in string");
      }
    }
    ++pos;
  }
  if (pos >= str.size()) {
    throw std::runtime_error("Unterminated string in JSON");
  }
  std::string result(str.substr(start, pos - start));
  ++pos;  // Skip closing quote
  return result;
}

/// @brief Expect and consume a specific character
/// @param[in] str The JSON string
/// @param[in,out] pos Current position (updated past the expected char)
/// @param[in] expected The character to expect
/// @throws std::runtime_error if character not found
inline auto expect_char(std::string_view str, size_t& pos, char expected)
    -> void {
  skip_whitespace(str, pos);
  if (pos >= str.size()) {
    throw std::runtime_error("Unexpected end of input, expected '" +
                             std::string(1, expected) + "'");
  }
  if (str[pos] != expected) {
    throw std::runtime_error("Expected '" + std::string(1, expected) +
                             "' but found '" + std::string(1, str[pos]) +
                             "' at position " + std::to_string(pos));
  }
  ++pos;
}

/// @brief Parse point coordinates [x, y] or [x, y, z]
/// @tparam Point Boost.Geometry point type
/// @param[in] str The JSON string containing coordinates
/// @param[in,out] pos Current position (updated past the coordinate array)
/// @return The parsed point
/// @throws std::runtime_error if coordinates are malformed
/// @note Z coordinate is parsed but ignored for 2D point types
template <typename Point>
[[nodiscard]] inline auto parse_point_coords(std::string_view str, size_t& pos)
    -> Point {
  Point pt;
  expect_char(str, pos, '[');
  skip_whitespace(str, pos);
  const double x = parse_number(str, pos);
  skip_whitespace(str, pos);
  expect_char(str, pos, ',');
  skip_whitespace(str, pos);
  const double y = parse_number(str, pos);
  skip_whitespace(str, pos);

  // Handle optional Z coordinate (skip it for 2D points)
  if (pos < str.size() && str[pos] == ',') {
    ++pos;
    skip_whitespace(str, pos);
    [[maybe_unused]] const double z = parse_number(str, pos);
    skip_whitespace(str, pos);
  }

  expect_char(str, pos, ']');

  boost::geometry::set<0>(pt, x);
  boost::geometry::set<1>(pt, y);
  return pt;
}

/// @brief Parse array of point coordinates [[x,y], [x,y], ...]
/// @tparam Point Boost.Geometry point type
/// @param[in] str The JSON string
/// @param[in,out] pos Current position
/// @return Vector of parsed points
template <typename Point>
[[nodiscard]] inline auto parse_point_array(std::string_view str, size_t& pos)
    -> std::vector<Point> {
  std::vector<Point> points;
  expect_char(str, pos, '[');
  skip_whitespace(str, pos);

  while (pos < str.size() && str[pos] != ']') {
    points.push_back(parse_point_coords<Point>(str, pos));
    skip_whitespace(str, pos);
    if (pos < str.size() && str[pos] == ',') {
      ++pos;
      skip_whitespace(str, pos);
    }
  }

  expect_char(str, pos, ']');
  return points;
}

/// @brief Parse array of linestring/ring coordinates [[[x,y],...], ...]
/// @tparam Point Boost.Geometry point type
/// @param[in] str The JSON string
/// @param[in,out] pos Current position
/// @return Vector of vectors of points (each inner vector is a linestring/ring)
template <typename Point>
[[nodiscard]] inline auto parse_linestring_array(std::string_view str,
                                                 size_t& pos)
    -> std::vector<std::vector<Point>> {
  std::vector<std::vector<Point>> linestrings;
  expect_char(str, pos, '[');
  skip_whitespace(str, pos);

  while (pos < str.size() && str[pos] != ']') {
    linestrings.push_back(parse_point_array<Point>(str, pos));
    skip_whitespace(str, pos);
    if (pos < str.size() && str[pos] == ',') {
      ++pos;
      skip_whitespace(str, pos);
    }
  }

  expect_char(str, pos, ']');
  return linestrings;
}

/// @brief Parse array of polygon coordinates [[[[x,y],...], ...], ...]
/// @tparam Point Boost.Geometry point type
/// @param[in] str The JSON string
/// @param[in,out] pos Current position
/// @return Vector of polygons (each polygon is a vector of rings)
template <typename Point>
[[nodiscard]] inline auto parse_polygon_array(std::string_view str, size_t& pos)
    -> std::vector<std::vector<std::vector<Point>>> {
  std::vector<std::vector<std::vector<Point>>> polygons;
  expect_char(str, pos, '[');
  skip_whitespace(str, pos);

  while (pos < str.size() && str[pos] != ']') {
    polygons.push_back(parse_linestring_array<Point>(str, pos));
    skip_whitespace(str, pos);
    if (pos < str.size() && str[pos] == ',') {
      ++pos;
      skip_whitespace(str, pos);
    }
  }

  expect_char(str, pos, ']');
  return polygons;
}

/// @brief Skip a JSON value (object, array, string, or primitive)
/// @param[in] str The JSON string
/// @param[in,out] pos Current position
/// @param[in] depth Current nesting depth (for overflow protection)
/// @throws std::runtime_error if nesting exceeds kMaxNestingDepth
inline auto skip_value(std::string_view str, size_t& pos, size_t depth = 0)
    -> void {
  if (depth > kMaxNestingDepth) {
    throw std::runtime_error("JSON nesting depth exceeded maximum of " +
                             std::to_string(kMaxNestingDepth));
  }
  skip_whitespace(str, pos);
  if (pos >= str.size()) {
    return;
  }

  const char c = str[pos];
  if (c == '"') {
    // Skip string - discard the returned value
    static_cast<void>(parse_string(str, pos));
  } else if (c == '[' || c == '{') {
    // Skip array or object recursively
    const char close = (c == '[') ? ']' : '}';
    ++pos;
    skip_whitespace(str, pos);
    while (pos < str.size() && str[pos] != close) {
      if (c == '{') {
        // Object: skip key
        skip_whitespace(str, pos);
        if (pos < str.size() && str[pos] == '"') {
          static_cast<void>(parse_string(str, pos));
          skip_whitespace(str, pos);
          if (pos < str.size() && str[pos] == ':') {
            ++pos;
          }
        }
      }
      skip_value(str, pos, depth + 1);
      skip_whitespace(str, pos);
      if (pos < str.size() && str[pos] == ',') {
        ++pos;
        skip_whitespace(str, pos);
      }
    }
    if (pos < str.size()) {
      ++pos;  // Skip closing bracket
    }
  } else {
    // Skip primitive value (number, boolean, null)
    while (pos < str.size() && str[pos] != ',' && str[pos] != '}' &&
           str[pos] != ']') {
      ++pos;
    }
  }
}

}  // namespace json

/// @brief Helper to parse a specific geometry type from GeoJSON
/// @tparam Geometry Boost.Geometry type to parse into
/// @param[in] geojson_str GeoJSON string (must be a valid geometry object)
/// @param[in,out] pos Current position in string (will be modified)
/// @return Parsed geometry
/// @throws std::runtime_error if coordinates field not found or malformed
template <typename Geometry>
[[nodiscard]] inline auto parse_geometry_from_geojson(
    std::string_view geojson_str, size_t& pos) -> Geometry {
  Geometry geom;
  bool found_coordinates = false;

  // Find coordinates field
  while (pos < geojson_str.size() && geojson_str[pos] != '}') {
    json::skip_whitespace(geojson_str, pos);
    if (pos >= geojson_str.size()) {
      break;
    }
    if (geojson_str[pos] == '"') {
      std::string key = json::parse_string(geojson_str, pos);
      json::skip_whitespace(geojson_str, pos);
      json::expect_char(geojson_str, pos, ':');
      json::skip_whitespace(geojson_str, pos);

      if (key == "coordinates") {
        found_coordinates = true;
        // Found coordinates, parse based on geometry type
        using Tag = typename boost::geometry::traits::tag<Geometry>::type;
        using Point = typename boost::geometry::point_type<Geometry>::type;

        if constexpr (std::is_same_v<Tag, boost::geometry::point_tag>) {
          geom = json::parse_point_coords<Geometry>(geojson_str, pos);
        } else if constexpr (std::is_same_v<Tag,
                                            boost::geometry::linestring_tag> ||
                             std::is_same_v<Tag, boost::geometry::ring_tag>) {
          auto points = json::parse_point_array<Point>(geojson_str, pos);
          boost::geometry::assign_points(geom, points);
        } else if constexpr (std::is_same_v<Tag,
                                            boost::geometry::polygon_tag>) {
          auto rings = json::parse_linestring_array<Point>(geojson_str, pos);
          if (!rings.empty()) {
            boost::geometry::assign_points(geom.outer(), rings[0]);
            for (size_t i = 1; i < rings.size(); ++i) {
              typename Geometry::ring_type inner;
              boost::geometry::assign_points(inner, rings[i]);
              geom.inners().push_back(std::move(inner));
            }
          }
        } else if constexpr (std::is_same_v<Tag,
                                            boost::geometry::multi_point_tag>) {
          auto points = json::parse_point_array<Point>(geojson_str, pos);
          for (const auto& pt : points) {
            geom.push_back(pt);
          }
        } else if constexpr (std::is_same_v<
                                 Tag, boost::geometry::multi_linestring_tag>) {
          auto linestrings =
              json::parse_linestring_array<Point>(geojson_str, pos);
          for (const auto& ls_points : linestrings) {
            pyinterp::geometry::LineString<Point> ls{ls_points};
            geom.push_back(ls);
          }
        } else if constexpr (std::is_same_v<
                                 Tag, boost::geometry::multi_polygon_tag>) {
          auto polygons = json::parse_polygon_array<Point>(geojson_str, pos);
          for (const auto& poly_rings : polygons) {
            pyinterp::geometry::Polygon<Point> poly;
            if (!poly_rings.empty()) {
              poly.outer() = pyinterp::geometry::Ring<Point>{poly_rings[0]};
              for (size_t i = 1; i < poly_rings.size(); ++i) {
                poly.inners().push_back(
                    pyinterp::geometry::Ring<Point>{poly_rings[i]});
              }
            }
            geom.push_back(poly);
          }
        }
        break;
      } else {
        // Skip other fields using the safe skip_value function
        json::skip_value(geojson_str, pos);
      }
    }
    json::skip_whitespace(geojson_str, pos);
    if (pos < geojson_str.size() && geojson_str[pos] == ',') {
      ++pos;
    }
  }

  if (!found_coordinates) {
    throw std::runtime_error(
        "GeoJSON object missing required 'coordinates' field");
  }

  return geom;
}

/// @brief Type traits to find geometry type by tag in parameter pack
/// @details Provides compile-time geometry type lookup based on boost geometry
/// tag. If no matching geometry is found, 'found' is false and 'type' is void.
template <typename TargetTag, typename... Geometries>
struct find_geometry_by_tag {
  static constexpr bool found = false;
  using type = void;  // Fallback type when not found
};

template <typename TargetTag, typename First, typename... Rest>
struct find_geometry_by_tag<TargetTag, First, Rest...> {
 private:
  using FirstTag = typename boost::geometry::traits::tag<First>::type;
  static constexpr bool is_match = std::is_same_v<FirstTag, TargetTag>;
  using RecursiveResult = find_geometry_by_tag<TargetTag, Rest...>;

 public:
  static constexpr bool found = is_match || RecursiveResult::found;
  using type =
      std::conditional_t<is_match, First, typename RecursiveResult::type>;
};

/// @brief Helper to define unified from_geojson using std::variant
/// @tparam Geometries All geometry types to support
/// @param[in] m Python module
/// @param[in] doc Documentation string
template <typename... Geometries>
inline auto define_from_geojson(nanobind::module_& m, const char* doc) -> void {
  using GeometryVariant = std::variant<Geometries...>;

  m.def(
      "from_geojson",
      [](const std::string& geojson_str) -> GeometryVariant {
        nanobind::gil_scoped_release release;

        if (geojson_str.empty()) {
          throw std::runtime_error("Empty GeoJSON string");
        }

        std::string_view str_view(geojson_str);
        size_t pos = 0;
        json::skip_whitespace(str_view, pos);
        json::expect_char(str_view, pos, '{');

        // First pass: find the "type" field to determine geometry type
        std::string geom_type;
        size_t type_search_pos = pos;
        while (type_search_pos < str_view.size() &&
               str_view[type_search_pos] != '}') {
          json::skip_whitespace(str_view, type_search_pos);
          if (type_search_pos >= str_view.size()) {
            break;
          }
          if (str_view[type_search_pos] == '"') {
            std::string key = json::parse_string(str_view, type_search_pos);
            json::skip_whitespace(str_view, type_search_pos);
            if (type_search_pos < str_view.size() &&
                str_view[type_search_pos] == ':') {
              ++type_search_pos;
              json::skip_whitespace(str_view, type_search_pos);
              if (key == "type" && type_search_pos < str_view.size() &&
                  str_view[type_search_pos] == '"') {
                geom_type = json::parse_string(str_view, type_search_pos);
                break;
              }
            }
          }
          ++type_search_pos;
        }

        if (geom_type.empty()) {
          throw std::runtime_error(
              "GeoJSON object missing required 'type' field");
        }

        // Parse based on detected type using type traits for robustness
        using PointType =
            typename find_geometry_by_tag<boost::geometry::point_tag,
                                          Geometries...>::type;
        using LineStringType =
            typename find_geometry_by_tag<boost::geometry::linestring_tag,
                                          Geometries...>::type;
        // Note: Ring is not a standard GeoJSON type; rings are represented as
        // LineStrings in GeoJSON format
        using PolygonType =
            typename find_geometry_by_tag<boost::geometry::polygon_tag,
                                          Geometries...>::type;
        using MultiPointType =
            typename find_geometry_by_tag<boost::geometry::multi_point_tag,
                                          Geometries...>::type;
        using MultiLineStringType =
            typename find_geometry_by_tag<boost::geometry::multi_linestring_tag,
                                          Geometries...>::type;
        using MultiPolygonType =
            typename find_geometry_by_tag<boost::geometry::multi_polygon_tag,
                                          Geometries...>::type;

        if (geom_type == "Point") {
          return parse_geometry_from_geojson<PointType>(str_view, pos);
        }
        if (geom_type == "LineString") {
          return parse_geometry_from_geojson<LineStringType>(str_view, pos);
        }
        if (geom_type == "Polygon") {
          return parse_geometry_from_geojson<PolygonType>(str_view, pos);
        }
        if (geom_type == "MultiPoint") {
          return parse_geometry_from_geojson<MultiPointType>(str_view, pos);
        }
        if (geom_type == "MultiLineString") {
          return parse_geometry_from_geojson<MultiLineStringType>(str_view,
                                                                  pos);
        }
        if (geom_type == "MultiPolygon") {
          return parse_geometry_from_geojson<MultiPolygonType>(str_view, pos);
        }

        throw std::runtime_error("Unknown or unsupported GeoJSON type: '" +
                                 geom_type +
                                 "'. Supported types: Point, LineString, "
                                 "Polygon, MultiPoint, MultiLineString, "
                                 "MultiPolygon");
      },
      "geojson"_a, doc);
}

/// @brief Initialize GeoJSON algorithms in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_geojson(nanobind::module_& m) -> void {
  if constexpr (NS == GeometryNamespace::kCartesian) {
    // to_geojson for all geometry types
    define_to_geojson<GEOJSON_TYPES(cartesian)>(m, kToGeojsonDoc);

    // from_geojson for all geometry types
    define_from_geojson<GEOJSON_TYPES(cartesian)>(m, kFromGeojsonDoc);
  } else {
    // to_geojson for all geometry types
    define_to_geojson<GEOJSON_TYPES(geographic)>(m, kToGeojsonDoc);

    // from_geojson for all geometry types
    define_from_geojson<GEOJSON_TYPES(geographic)>(m, kFromGeojsonDoc);
  }
}

}  // namespace pyinterp::geometry::pybind
