// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/fill/fft_inpaint.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include "pyinterp/eigen.hpp"
#include "pyinterp/fill/fft_inpaint.hpp"

namespace nb = nanobind;

namespace pyinterp::fill::pybind {

constexpr const char* const kFFTInpaintDocstring = R"(
Fill undefined values using spectral in-painting (FFT/DCT-based).

Spectral in-painting uses iterative Gaussian filtering in frequency space to
fill NaN values. For periodic grids, uses FFT; for non-periodic grids, uses
DCT (which implicitly handles symmetric boundary conditions).

Args:
    grid: Input 2D array with data to fill. NaN values mark undefined points.
    config: FFT Inpaint configuration with:
        - first_guess: Initial guess strategy (zonal_average or zero)
        - is_periodic: Whether the x-axis is periodic (e.g., longitude)
        - max_iterations: Maximum number of smoothing iterations
        - epsilon: Convergence threshold for residual
        - sigma: Gaussian filter parameter controlling frequency cutoff
        - num_threads: Number of threads (0 = use all available CPUs)

Returns:
    Tuple of (iterations, max_residual):
        - iterations: Number of iterations performed
        - max_residual: Maximum residual value at convergence
)";

/// Bind FFT Inpaint function to Python module.
/// @tparam T Scalar type
/// @param m Python module
template <std::floating_point T>
void bind_fft_inpaint(nanobind::module_& m) {
  m.def(
      "fft_inpaint",
      [](const Eigen::Ref<RowMajorMatrix<T>>& grid,
         const config::fill::FFTInpaint& config) -> std::tuple<size_t, T> {
        return fft_inpaint(grid, config);
      },
      nanobind::arg("grid"), nanobind::arg("config"), kFFTInpaintDocstring,
      nanobind::call_guard<nanobind::gil_scoped_release>());
}

void bind_fft_inpaint(nanobind::module_& m) {
  bind_fft_inpaint<double>(m);
  bind_fft_inpaint<float>(m);
}

}  // namespace pyinterp::fill::pybind
