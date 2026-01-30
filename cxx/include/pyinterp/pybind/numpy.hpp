// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <Eigen/Core>

#include "pyinterp/dateutils.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp::pybind {

/// @brief Singleton context for numpy operations
struct NumpyContext {
  nanobind::object module;  /// Reference to numpy module

  /// @brief Get the singleton instance
  /// @return The singleton instance
  ///
  /// @note Uses intentional memory leak to avoid destructor issues during
  /// Python interpreter shutdown. The module reference is never destroyed,
  /// which is safe since it's a process-lifetime singleton.
  static auto get() -> const NumpyContext & {
    static auto *ctx = new NumpyContext{nanobind::module_::import_("numpy")};
    return *ctx;
  }
};

/// @brief Convert and validate the dtype of a numpy datetime64/timedelta64
/// array
/// @param[in] name Variable name
/// @param[in] array Numpy array to validate
/// @return The datetime64 or timedelta64 dtype of the array
inline auto retrieve_dtype(const std::string &name,
                           const nanobind::object &array) -> dateutils::DType {
  nanobind::object arr_dtype = array.attr("dtype");
  auto arr_kind = nanobind::cast<std::string>(nanobind::str(arr_dtype));
  try {
    return dateutils::DType(arr_kind);
  } catch (const std::invalid_argument &) {
    throw std::invalid_argument(
        name + " must be a numpy.datetime64 or numpy.timedelta64 array, got " +
        arr_kind);
  }
}
/// @brief Convert a numpy datetime64/timedelta64 array to an Eigen vector of
/// int64_t values
/// @param[in] array Numpy array to convert
/// @return Eigen vector of int64_t values
inline auto numpy_to_vector(const nanobind::object &array) -> Vector<int64_t> {
  // Accept read-only arrays (e.g., from xarray) by specifying nanobind::ro
  auto viewed =
      nanobind::cast<nanobind::ndarray<nanobind::numpy, nanobind::ro, int64_t,
                                       nanobind::ndim<1>, nanobind::c_contig>>(
          array.attr("view")("int64"));
  return Eigen::Map<const Vector<int64_t>>(
      viewed.data(), static_cast<int64_t>(viewed.shape(0)));
}

/// @brief Convert a numpy datetime64/timedelta64 array to an Eigen matrix of
/// int64_t values
/// @param[in] array Numpy array to convert
/// @return Eigen matrix of int64_t values
inline auto numpy_to_matrix(const nanobind::object &array) -> Matrix<int64_t> {
  // Accept read-only arrays (e.g., from xarray) by specifying nanobind::ro
  auto viewed =
      nanobind::cast<nanobind::ndarray<nanobind::numpy, nanobind::ro, int64_t,
                                       nanobind::ndim<2>, nanobind::c_contig>>(
          array.attr("view")("int64"));
  return Eigen::Map<const Matrix<int64_t>>(
      viewed.data(), static_cast<int64_t>(viewed.shape(0)),
      static_cast<int64_t>(viewed.shape(1)));
}

/// @brief Return a datetime64/timedelta64 numpy array from an Eigen vector of
/// int64_t values
/// @param[in] vector Eigen vector of int64_t values
/// @param[in] dtype Target numpy dtype (datetime64 or timedelta64)
/// @return Numpy array of datetime64/timedelta64 values
inline auto vector_to_numpy(Vector<int64_t> &&vector, dateutils::DType dtype)
    -> nanobind::object {
  // Create an int64 ndarray
  auto size = static_cast<size_t>(vector.size());
  auto ptr = std::make_unique<Vector<int64_t>>(std::move(vector));

  auto *data = ptr->data();
  nanobind::capsule capsule(ptr.get(), [](void *data) noexcept {
    delete static_cast<Vector<int64_t> *>(data);
  });
  ptr.release();

  nanobind::ndarray<nanobind::numpy, int64_t, nanobind::ndim<1>> arr(
      data, {size}, capsule);

  // Convert to numpy object and view as datetime64/timedelta64
  return arr.cast().attr("view")(std::string(dtype));
}

/// @brief Return the numpy dtype object from a dateutils::DType
/// @param[in] dtype The dateutils::DType object
/// @return The corresponding numpy dtype object
inline auto to_dtype(const dateutils::DType &dtype) -> nanobind::object {
  auto np = NumpyContext::get().module;
  return np.attr("dtype")(std::string(dtype));
}

/// @brief Create a datetime64 scalar from a 64-bit integer value
/// @param[in] value Integer value representing the datetime64
/// @param[in] dtype Target numpy dtype
/// @return Numpy datetime64 scalar
inline auto make_datetime64_scalar(int64_t value, dateutils::DType dtype)
    -> nanobind::object {
  auto np = NumpyContext::get().module;
  auto datetime64_attr = np.attr("datetime64");
  return datetime64_attr(value, dtype.unit().data());
}

/// @brief Create a timedelta64 scalar from a 64-bit integer value
/// @param[in] value Integer value representing the timedelta64
/// @param[in] dtype Target numpy dtype
/// @return Numpy timedelta64 scalar
inline auto make_timedelta64_scalar(int64_t value, dateutils::DType dtype)
    -> nanobind::object {
  auto np = NumpyContext::get().module;
  auto timedelta64_attr = np.attr("timedelta64");
  return timedelta64_attr(value, dtype.unit().data());
}

/// @brief Create a datetime64/timedelta64 scalar from a 64-bit integer value
/// @param[in] value Integer value representing the datetime64 or timedelta64
/// @param[in] dtype Target numpy dtype
/// @return Numpy datetime64 or timedelta64 scalar
inline auto make_scalar(int64_t value, dateutils::DType dtype)
    -> nanobind::object {
  return (dtype.datetype() == dateutils::DType::DateType::kDatetime64)
             ? make_datetime64_scalar(value, dtype)
             : make_timedelta64_scalar(value, dtype);
}

}  // namespace pyinterp::pybind
