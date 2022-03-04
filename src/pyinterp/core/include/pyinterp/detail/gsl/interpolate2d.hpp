// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <gsl/gsl_spline2d.h>

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "pyinterp/detail/gsl/accelerator.hpp"

namespace pyinterp::detail::gsl {

/// Interpolate a 1-D function
class Interpolate2D {
 public:
  /// Interpolate a 1-D function
  ///
  /// @param xsize grid points in the X direction
  /// @param ysize grid points in the Y direction
  /// @param type fitting model
  /// @param acc Accelerator
  Interpolate2D(const size_t xsize, const size_t ysize,
                const gsl_interp2d_type *type, Accelerator xacc,
                Accelerator yacc)
      : workspace_(
            std::unique_ptr<gsl_spline2d, std::function<void(gsl_spline2d *)>>(
                gsl_spline2d_alloc(type, xsize, ysize),
                [](gsl_spline2d *ptr) { gsl_spline2d_free(ptr); })),
        xacc_(std::move(xacc)),
        yacc_(std::move(yacc)) {}

  /// Returns the name of the interpolation type used
  [[nodiscard]] inline auto name() const noexcept -> std::string {
    return gsl_spline2d_name(workspace_.get());
  }

  /// Return the minimum number of points required by the interpolation
  [[nodiscard]] inline auto min_size() const noexcept -> size_t {
    return gsl_spline2d_min_size(workspace_.get());
  }

  /// Return the interpolated value of z for a given point x, y
  [[nodiscard]] inline auto evaluate(const Eigen::VectorXd &xa,
                                     const Eigen::VectorXd &ya,
                                     const Eigen::MatrixXd &za, const double x,
                                     const double y) -> double {
    init(xa, ya, za);
    return gsl_spline2d_eval(workspace_.get(), x, y, xacc_, yacc_);
  }

 private:
  std::unique_ptr<gsl_spline2d, std::function<void(gsl_spline2d *)>> workspace_;
  Accelerator xacc_;
  Accelerator yacc_;

  /// Initializes the interpolation object
  inline auto init(const Eigen::VectorXd &xa, const Eigen::VectorXd &ya,
                   const Eigen::MatrixXd &za) noexcept -> void {
    xacc_.reset();
    xacc_.reset();
    gsl_spline2d_init(workspace_.get(), xa.data(), ya.data(), za.data(),
                      xa.size(), ya.size());
  }
};

}  // namespace pyinterp::detail::gsl
