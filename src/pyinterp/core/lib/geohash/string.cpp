// Copyright (c) 2020 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geohash/string.hpp"

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geodetic/box.hpp"
#include "pyinterp/geohash/base32.hpp"
#include "pyinterp/geohash/int64.hpp"

namespace pyinterp::geohash::string {

// Handle encoding/decoding in base32
static const auto base32 = Base32();

// ---------------------------------------------------------------------------
auto Array::get_info(const pybind11::array &hash, const pybind11::ssize_t ndim)
    -> pybind11::buffer_info {
  auto info = hash.request();
  auto dtype = hash.dtype();
  if ((hash.flags() & pybind11::array::c_style) == 0) {
    throw std::runtime_error("hash must be C-style contiguous");
  }
  switch (ndim) {
    case 1:
      if (info.ndim != 1) {
        throw std::invalid_argument("hash must be a one-dimensional array");
      }
      if (dtype.kind() != 'S') {
        throw std::invalid_argument("hash must be a string array");
      }
      if (info.strides[0] > 12) {
        throw std::invalid_argument("hash length must be within [1, 12]");
      }
      break;
    default:
      if (info.ndim != 2) {
        throw std::invalid_argument("hash must be a two-dimensional array");
      }
      if (info.strides[0] != hash.shape(1) * info.strides[1] ||
          dtype.kind() != 'S') {
        throw std::invalid_argument("hash must be a string array");
      }
      if (info.strides[1] > 12) {
        throw std::invalid_argument("hash length must be within [1, 12]");
      }
      break;
  }
  return info;
}

// ---------------------------------------------------------------------------
auto allocate_array(const size_t size, const uint32_t precision) -> Array {
  try {
    try {
      return {size, precision};
    } catch (const std::length_error &) {
      throw std::bad_alloc();
    }
  } catch (const std::bad_alloc &) {
    auto ss = std::stringstream();
    ss << "Unable to allocate " << int64::format_bytes(size * precision)
       << " for an array with shape (" << size << ",) and data type S"
       << precision;
    PyErr_SetString(PyExc_MemoryError, ss.str().c_str());
    throw pybind11::error_already_set();
  }
}

// ---------------------------------------------------------------------------
auto encode(const geodetic::Point &point, char *const buffer,
            const uint32_t precision) -> void {
  Base32::encode(int64::encode(point, 5 * precision), buffer, precision);
}

// ---------------------------------------------------------------------------
auto encode(const Eigen::Ref<const Eigen::VectorXd> &lon,
            const Eigen::Ref<const Eigen::VectorXd> &lat,
            const uint32_t precision) -> pybind11::array {
  detail::check_eigen_shape("lon", lon, "lat", lat);
  auto size = lon.size();
  auto array = allocate_array(size, precision);
  auto *buffer = array.buffer();

  {
    auto gil = pybind11::gil_scoped_release();

    for (Eigen::Index ix = 0; ix < size; ++ix) {
      encode({lon[ix], lat[ix]}, buffer, precision);
      buffer += precision;
    }
  }
  return array.pyarray();
}

// ---------------------------------------------------------------------------
inline auto decode_bounding_box(const char *const hash, const size_t count,
                                uint32_t *precision = nullptr)
    -> geodetic::Box {
  auto [integer_encoded, chars] = base32.decode(hash, count);
  if (precision != nullptr) {
    *precision = chars;
  }
  return int64::bounding_box(integer_encoded, 5 * chars);
}

// ---------------------------------------------------------------------------
auto bounding_box(const char *const hash, const size_t count) -> geodetic::Box {
  return decode_bounding_box(hash, count);
}

// ---------------------------------------------------------------------------
auto decode(const char *const hash, const size_t count, const bool round)
    -> geodetic::Point {
  auto bbox = bounding_box(hash, count);
  return round ? bbox.round() : bbox.centroid();
}

// ---------------------------------------------------------------------------
auto decode(const pybind11::array &hash, const bool round)
    -> std::tuple<Eigen::VectorXd, Eigen::VectorXd> {
  auto info = Array::get_info(hash, 1);
  auto count = info.strides[0];
  auto lon = Eigen::VectorXd(info.shape[0]);
  auto lat = Eigen::VectorXd(info.shape[0]);
  auto *ptr = static_cast<char *>(info.ptr);
  {
    auto gil = pybind11::gil_scoped_release();
    for (auto ix = 0LL; ix < info.shape[0]; ++ix) {
      auto point = decode(ptr, count, round);
      lon[ix] = point.lon();
      lat[ix] = point.lat();
      ptr += count;
    }
  }
  return std::make_tuple(lon, lat);
}

// ---------------------------------------------------------------------------
auto neighbors(const char *const hash, const size_t count) -> pybind11::array {
  auto [integer_encoded, precision] = base32.decode(hash, count);

  const auto integers = int64::neighbors(integer_encoded, precision * 5);
  auto array = allocate_array(integers.size(), precision);
  auto *buffer = array.buffer();

  {
    auto gil = pybind11::gil_scoped_release();
    for (auto ix = 0; ix < integers.size(); ++ix) {
      Base32::encode(integers(ix), buffer, precision);
      buffer += precision;
    }
  }
  return array.pyarray();
}

// ---------------------------------------------------------------------------
auto area(const pybind11::array &hash,
          const std::optional<geodetic::System> &wgs) -> Eigen::MatrixXd {
  auto info = Array::get_info(hash, 1);
  auto count = info.strides[0];
  auto result = Eigen::VectorXd(info.shape[0]);
  auto *ptr = static_cast<char *>(info.ptr);
  auto spheroid = wgs.has_value()
                      ? boost::geometry::srs::spheroid(wgs->semi_major_axis(),
                                                       wgs->semi_minor_axis())
                      : boost::geometry::srs::spheroid<double>();
  auto strategy = boost::geometry::strategy::area::geographic<
      boost::geometry::strategy::vincenty, 5>(spheroid);
  {
    auto gil = pybind11::gil_scoped_release();
    for (auto ix = 0LL; ix < info.shape[0]; ++ix) {
      result[ix] = boost::geometry::area(
          static_cast<geodetic::Polygon>(bounding_box(ptr, count)), strategy);
      ptr += count;
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
auto bounding_boxes(const std::optional<geodetic::Box> &box,
                    const uint32_t precision) -> pybind11::array {
  // Number of bits
  auto bits = precision * 5;

  // If the input bbox cut the meridian, we need to split it into two
  // bounding boxes.
  const auto boxes =
      box.value_or(geodetic::Box::whole_earth()).normalize().split();

  // Grid resolution in degrees
  const auto lng_lat_err = int64::error_with_precision(bits);

  // Allocation of the vector storing the different codes of the matrix created
  auto result = allocate_array(int64::count(boxes, bits), precision);
  auto *buffer = result.buffer();

  {
    auto gil = pybind11::gil_scoped_release();

    for (const auto &item : boxes) {
      auto [hash_sw, lon_step, lat_step] = int64::grid_properties(item, bits);
      const auto point_sw = int64::decode(hash_sw, bits, true);

      for (size_t lat = 0; lat < lat_step; ++lat) {
        const auto lat_shift =
            static_cast<double>(lat) * std::get<1>(lng_lat_err);

        for (size_t lon = 0; lon < lon_step; ++lon) {
          const auto lon_shift =
              static_cast<double>(lon) * std::get<0>(lng_lat_err);

          Base32::encode(int64::encode({point_sw.lon() + lon_shift,
                                        point_sw.lat() + lat_shift},
                                       bits),
                         buffer, precision);
          buffer += precision;
        }
      }
    }
  }
  return result.pyarray();
}

// ---------------------------------------------------------------------------
// Calculates a grid containing for each cell a boolean indicating if the cell
// of the grid is enclosed or not in the polygon.
static auto mask_box(const geodetic::Box &box, const geodetic::Polygon &polygon,
                     const std::tuple<double, double> &lng_lat_err,
                     const uint32_t bits, const size_t num_threads)
    -> Matrix<bool> {
  size_t lat_step;
  size_t lon_step;
  uint64_t hash_sw;

  // Calculate the grid properties
  std::tie(hash_sw, lon_step, lat_step) = int64::grid_properties(box, bits);
  const auto point_sw = int64::decode(hash_sw, bits, true);

  // Allocate the grid result
  auto result = Matrix<bool>(lon_step, lat_step);

  // Captures the detected exceptions in the calculation function
  // (only the last exception captured is kept)
  auto except = std::exception_ptr(nullptr);

  detail::dispatch(
      [&](size_t start, size_t end) {
        for (auto lat = static_cast<int64_t>(start);
             lat < static_cast<int64_t>(end); ++lat) {
          const auto lat_shift =
              static_cast<double>(lat) * std::get<1>(lng_lat_err);

          for (size_t lon = 0; lon < lon_step; ++lon) {
            const auto lon_shift =
                static_cast<double>(lon) * std::get<0>(lng_lat_err);

            result(lon, lat) = boost::geometry::intersects(
                int64::bounding_box(
                    int64::encode(geodetic::Point(point_sw.lon() + lon_shift,
                                                  point_sw.lat() + lat_shift),
                                  bits),
                    bits),
                polygon);
          }
        }
      },
      lat_step, num_threads);

  if (except != nullptr) {
    std::rethrow_exception(except);
  }

  return result;
}

// ---------------------------------------------------------------------------
static auto mask_boxes(const std::vector<geodetic::Box> &boxes,
                       const geodetic::Polygon &polygon,
                       const std::tuple<double, double> &lng_lat_err,
                       const uint32_t precision, const size_t num_threads)
    -> std::vector<Matrix<bool>> {
  // Allocation of the vector storing the different codes of the matrix created
  auto result = std::vector<Matrix<bool>>();

  {
    auto gil = pybind11::gil_scoped_release();

    for (const auto &item : boxes) {
      result.emplace_back(
          mask_box(item, polygon, lng_lat_err, precision, num_threads));
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
auto bounding_boxes(const geodetic::Polygon &polygon, const uint32_t precision,
                    const size_t num_threads) -> pybind11::array {
  // Number of bits
  auto bits = precision * 5;

  // Bounding box of the grid to be created
  const auto envelope = polygon.envelope();

  // If the envelope cut the meridian, we need to split it into two
  // bounding boxes.
  auto split_box = [](const geodetic::Box &box) -> std::vector<geodetic::Box> {
    auto boxes = box.normalize().split();
    return {std::make_move_iterator(boxes.begin()),
            std::make_move_iterator(boxes.end())};
  };
  auto boxes = split_box(envelope);

  // Grid resolution in degrees
  const auto lng_lat_err = int64::error_with_precision(bits);

  // Calculates the intersection mask between the polygon and the GeoHash grid
  // (multithreaded)
  auto masks = mask_boxes(boxes, polygon, lng_lat_err, bits, num_threads);

  // Count the number of cells that are enclosed by the polygon
  auto size = size_t(0);
  for (const auto &item : masks) {
    size += static_cast<size_t>(
        std::count(item.data(), item.data() + item.size(), true));
  }

  // Allocates the result array
  auto result = allocate_array(size, precision);
  auto *buffer = result.buffer();

  // Finally, for each cell of the grid, if it is enclosed by the polygon,
  // we add the code to the result array
  {
    auto gil = pybind11::gil_scoped_release();

    for (auto ix = size_t(0); ix < masks.size(); ++ix) {
      auto box = boxes[ix];
      auto mask = masks[ix];

      auto [hash_sw, lon_step, lat_step] = int64::grid_properties(box, bits);
      const auto point_sw = int64::decode(hash_sw, bits, true);
      for (size_t lat = 0; lat < lat_step; ++lat) {
        const auto lat_shift =
            static_cast<double>(lat) * std::get<1>(lng_lat_err);

        for (size_t lon = 0; lon < lon_step; ++lon) {
          if (mask(lon, lat)) {
            const auto lon_shift =
                static_cast<double>(lon) * std::get<0>(lng_lat_err);

            Base32::encode(
                int64::encode(geodetic::Point(point_sw.lon() + lon_shift,
                                              point_sw.lat() + lat_shift),
                              bits),
                buffer, precision);
            buffer += precision;
            ++size;
          }
        }
      }
    }
  }
  return result.pyarray();
}

// ---------------------------------------------------------------------------
auto where(const pybind11::array &hash) -> std::unordered_map<
    std::string,
    std::tuple<std::tuple<int64_t, int64_t>, std::tuple<int64_t, int64_t>>> {
  // Index shifts of neighboring pixels
  static const auto shift_row =
      std::array<int64_t, 8>{-1, -1, -1, 0, 1, 0, 1, 1};
  static const auto shift_col =
      std::array<int64_t, 8>{-1, 1, 0, -1, -1, 1, 0, 1};

  auto result = std::unordered_map<
      std::string,
      std::tuple<std::tuple<int64_t, int64_t>, std::tuple<int64_t, int64_t>>>();

  auto info = Array::get_info(hash, 2);
  auto rows = info.shape[0];
  auto cols = info.shape[1];
  auto chars = info.strides[1];
  auto *ptr = static_cast<char *>(info.ptr);
  std::string current_code;
  std::string neighboring_code;

  assert(chars <= 12);

  {
    auto gil = pybind11::gil_scoped_release();

    for (int64_t ix = 0; ix < rows; ++ix) {
      for (int64_t jx = 0; jx < cols; ++jx) {
        current_code = std::string(ptr + (ix * cols + jx) * chars, chars);

        auto it = result.find(current_code);
        if (it == result.end()) {
          result.emplace(std::make_pair(
              current_code, std::make_tuple(std::make_tuple(ix, ix),
                                            std::make_tuple(jx, jx))));
          continue;
        }

        for (int64_t kx = 0; kx < 8; ++kx) {
          const auto i = ix + shift_row[kx];
          const auto j = jx + shift_col[kx];

          if (i >= 0 && i < rows && j >= 0 && j < cols) {
            neighboring_code = std::string(ptr + (i * cols + j) * chars, chars);
            if (current_code == neighboring_code) {
              auto &first = std::get<0>(it->second);
              std::get<0>(first) = std::min(std::get<0>(first), i);
              std::get<1>(first) = std::max(std::get<1>(first), i);

              auto &second = std::get<1>(it->second);
              std::get<0>(second) = std::min(std::get<0>(second), j);
              std::get<1>(second) = std::max(std::get<1>(second), j);
            }
          }
        }
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
static auto zoom_in(char *ptr, pybind11::ssize_t size, uint32_t from_precision,
                    uint32_t to_precision) -> pybind11::array {
  // Number of bits need to zoom in
  auto bits = to_precision * 5;

  // Calculation of the number of items needed for the result.
  auto size_in =
      size * (size_t(2) << (5 * (to_precision - from_precision) - 1));

  // Allocates the result table.
  auto result = allocate_array(size_in, to_precision);
  auto *buffer = result.buffer();

  {
    auto gil = pybind11::gil_scoped_release();

    for (auto ix = 0; ix < size; ++ix) {
      auto codes =
          int64::bounding_boxes(bounding_box(ptr, from_precision), bits);
      for (auto jx = 0; jx < codes.size(); ++jx) {
        Base32::encode(codes[jx], buffer, to_precision);
        buffer += to_precision;
      }
      ptr += from_precision;
    }
  }
  return result.pyarray();
}

// ---------------------------------------------------------------------------
static auto zoom_out(char *ptr, pybind11::ssize_t size, uint32_t from_precision,
                     uint32_t to_precision) -> pybind11::array {
  auto current_code = std::string(to_precision, '\0');
  auto zoom_out_codes = std::set<std::string>();

  {
    auto gil = pybind11::gil_scoped_release();

    for (auto ix = 0; ix < size; ++ix) {
      encode(decode(ptr, from_precision, false), current_code.data(),
             to_precision);
      zoom_out_codes.emplace(std::string(current_code));
      ptr += from_precision;
    }
  }

  auto result = allocate_array(zoom_out_codes.size(), to_precision);
  auto *buffer = result.buffer();
  for (auto code : zoom_out_codes) {
    std::copy(code.begin(), code.end(), buffer);
    buffer += to_precision;
  }

  return result.pyarray();
}

// ---------------------------------------------------------------------------
auto transform(const pybind11::array &hash, uint32_t precision)
    -> pybind11::array {
  // Decode the information in the provided table.
  auto info = Array::get_info(hash, 1);
  auto size = info.shape[0];
  auto input_precision = static_cast<uint32_t>(info.strides[0]);
  auto *ptr = static_cast<char *>(info.ptr);

  if (input_precision == precision) {
    return hash;
  }
  if (input_precision > precision) {
    return zoom_out(ptr, size, input_precision, precision);
  }
  return zoom_in(ptr, size, input_precision, precision);
}

}  // namespace pyinterp::geohash::string
