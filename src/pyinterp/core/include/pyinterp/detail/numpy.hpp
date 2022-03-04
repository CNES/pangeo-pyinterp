// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>

#include <list>
#include <tuple>
#include <vector>

#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::numpy {

/// Checks if the given axes are valid for the given array.
template <typename T>
void check_axis_bounds(
    const pybind11::array_t<T, pybind11::array::c_style> &arr,
    const std::list<pybind11::ssize_t> &axis) {
  auto ndim = arr.ndim();
  auto [min, max] = std::minmax_element(axis.begin(), axis.end());
  if (*min < 0) {
    throw std::invalid_argument("axis " + std::to_string(*min) +
                                " is out of bounds for array of dimension " +
                                std::to_string(ndim));
  }
  if (*max > ndim - 1) {
    throw std::invalid_argument("axis " + std::to_string(*max) +
                                " is out of bounds for array of dimension " +
                                std::to_string(ndim));
  }
}

/// Converts a flat index into a vector of indices for each dimension of the
/// tensor.
inline auto unravel(pybind11::ssize_t index,
                    const Vector<pybind11::ssize_t> &strides,
                    Vector<pybind11::ssize_t> &indexes) -> void {
  for (pybind11::ssize_t ix = 0; ix < strides.size(); ++ix) {
    indexes[ix] = index / strides[ix];
    index -= indexes[ix] * strides[ix];
  }
}

/// Get the properties of the tensors handled to perform the reduction.
template <typename T>
[[nodiscard]] auto reduced_properties(
    const pybind11::array_t<T, pybind11::array::c_style> &arr,
    const std::list<pybind11::ssize_t> &axes)
    -> std::tuple<std::vector<pybind11::ssize_t>, Vector<pybind11::ssize_t>,
                  Vector<pybind11::ssize_t>> {
  // Properties of the input tensor
  auto ndim = arr.ndim();
  auto strides = Vector<pybind11::ssize_t>(ndim);

  // Reduced tensor shape
  auto reduced_shape = std::vector<pybind11::ssize_t>();

  // Preserved dimensions of the processed tensor.
  auto adjusted_dims = Vector<pybind11::ssize_t>(ndim + 1);

  // Preserved strides of the processed tensor.
  auto adjusted_strides = Vector<pybind11::ssize_t>(ndim);

  // Calculation of the shape of the reduced tensor.
  for (pybind11::ssize_t ix = 0; ix < arr.ndim(); ++ix) {
    auto dim = arr.shape(ix);
    strides[ix] = arr.strides(ix) / sizeof(T);

    // We keep this dimension?
    if (std::find(axes.begin(), axes.end(), ix) == axes.end()) {
      reduced_shape.push_back(dim);
      adjusted_dims[ix] = dim;
    } else {
      adjusted_dims[ix] = -1;
    }
  }

  // To handle automatically the last stride of the tensor
  adjusted_dims[ndim] = 1;

  // Calculate the preserved strides of the processed tensor.
  for (pybind11::ssize_t ix = 0; ix < ndim; ++ix) {
    if (adjusted_dims[ix] == -1) {
      adjusted_strides[ix] = 0;
    } else {
      adjusted_strides[ix] = std::abs(adjusted_dims.tail(ndim - ix).prod());
    }
  }
  return {reduced_shape, strides, adjusted_strides};
}

/// Returns an array of ones with the same shape as a given array.
template <typename T>
[[nodiscard]] auto ones_like(
    pybind11::array_t<T, pybind11::array::c_style> &values)
    -> pybind11::array_t<T, pybind11::array::c_style> {
  auto ones = pybind11::array_t<T, pybind11::array::c_style>(
      pybind11::array::ShapeContainer{values.size()});
  auto ptr_ones =
      reinterpret_cast<T *>(pybind11::detail::array_proxy(ones.ptr())->data);
  std::fill(ptr_ones, ptr_ones + values.size(), T(1));
  return ones;
}

template <typename T>
[[nodiscard]] constexpr auto get_data_pointer(void *ptr) -> T * {
  return reinterpret_cast<T *>(pybind11::detail::array_proxy(ptr)->data);
}

}  // namespace pyinterp::detail::numpy
