// Copyright (c) 2020 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <gsl/gsl_interp2d.h>

#include <Eigen/Core>
#include <functional>
#include <memory>

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
                const gsl_interp2d_type* type, Accelerator xacc,
                Accelerator yacc)
      : workspace_(
            std::unique_ptr<gsl_interp2d, std::function<void(gsl_interp2d*)>>(
                gsl_interp2d_alloc(type, xsize, ysize),
                [](gsl_interp2d* ptr) { gsl_interp2d_free(ptr); })),
        xacc_(std::move(xacc)),
        yacc_(std::move(yacc)) {}

  /// Returns the name of the interpolation type used
  [[nodiscard]] inline auto name() const noexcept -> std::string {
    return gsl_interp2d_name(workspace_.get());
  }

  /// Return the minimum number of points required by the interpolation
  [[nodiscard]] inline auto min_size() const noexcept -> size_t {
    return gsl_interp2d_min_size(workspace_.get());
  }

  /// Return the interpolated value of z for a given point x, y
  [[nodiscard]] inline auto interpolate(const Eigen::VectorXd& xa,
                                        const Eigen::VectorXd& ya,
                                        const Eigen::VectorXd& za,
                                        const double x, const double y)
      -> double {
    init(xa, ya, za);
    return gsl_interp2d_eval(workspace_.get(), xa.data(), ya.data(), za.data(),
                             x, y, xacc_, yacc_);
  }

 private:
  std::unique_ptr<gsl_interp2d, std::function<void(gsl_interp2d*)>> workspace_;
  Accelerator xacc_;
  Accelerator yacc_;

  /// Initializes the interpolation object
  inline void init(const Eigen::VectorXd& xa, const Eigen::VectorXd& ya,
                   const Eigen::VectorXd& za) noexcept {
    xacc_.reset();
    xacc_.reset();
    gsl_interp2d_init(workspace_.get(), xa.data(), ya.data(), za.data(),
                      xa.size(), ya.size());
  }
};

}  // namespace pyinterp::detail::gsl