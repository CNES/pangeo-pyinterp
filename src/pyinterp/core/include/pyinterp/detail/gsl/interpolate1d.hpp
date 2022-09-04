// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <gsl/gsl_spline.h>

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "pyinterp/detail/gsl/accelerator.hpp"

namespace pyinterp::detail::gsl {

/// Interpolate a 1-D function
class Interpolate1D {
 public:
  /// Interpolate a 1-D function
  ///
  /// @param size Size of workspace
  /// @param type fitting model
  /// @param acc Accelerator
  Interpolate1D(const size_t size, const gsl_interp_type *type, Accelerator acc)
      : workspace_(
            std::unique_ptr<gsl_spline, std::function<void(gsl_spline *)>>(
                gsl_spline_alloc(type, size),
                [](gsl_spline *ptr) { gsl_spline_free(ptr); })),
        acc_(std::move(acc)) {}

  /// Returns the name of the interpolation type used
  [[nodiscard]] inline auto name() const noexcept -> std::string {
    return gsl_spline_name(workspace_.get());
  }

  /// Return the minimum number of points required by the interpolation
  [[nodiscard]] inline auto min_size() const noexcept -> size_t {
    return gsl_spline_min_size(workspace_.get());
  }

  /// Return the interpolated value of y for a given point x
  [[nodiscard]] inline auto interpolate(const Eigen::VectorXd &xa,
                                        const Eigen::VectorXd &ya,
                                        const double x) -> double {
    init(xa, ya);
    return gsl_spline_eval(workspace_.get(), x, acc_);
  }

  /// Return the derivative d of an interpolated function for a given point x
  [[nodiscard]] inline auto derivative(const Eigen::VectorXd &xa,
                                       const Eigen::VectorXd &ya,
                                       const double x) -> double {
    init(xa, ya);
    return gsl_spline_eval_deriv(workspace_.get(), x, acc_);
  }

  /// Return the second derivative d of an interpolated function for a given
  /// point x
  [[nodiscard]] inline auto second_derivative(const Eigen::VectorXd &xa,
                                              const Eigen::VectorXd &ya,
                                              const double x) -> double {
    init(xa, ya);
    return gsl_spline_eval_deriv2(workspace_.get(), x, acc_);
  }

  /// Return the numerical integral result of an interpolated function over the
  /// range [a, b],
  [[nodiscard]] inline auto integral(const Eigen::VectorXd &xa,
                                     const Eigen::VectorXd &ya, const double a,
                                     const double b) -> double {
    init(xa, ya);
    return gsl_spline_eval_integ(workspace_.get(), a, b, acc_);
  }

  static inline auto parse_interp_type(const std::string &kind)
      -> const gsl_interp_type * {
    if (kind == "linear") {
      return gsl_interp_linear;
    }
    if (kind == "polynomial") {
      return gsl_interp_polynomial;
    }
    if (kind == "c_spline") {
      return gsl_interp_cspline;
    }
    if (kind == "c_spline_periodic") {
      return gsl_interp_cspline_periodic;
    }
    if (kind == "akima") {
      return gsl_interp_akima;
    }
    if (kind == "akima_periodic") {
      return gsl_interp_akima_periodic;
    }
    if (kind == "steffen") {
      return gsl_interp_steffen;
    }
    throw std::invalid_argument("Invalid spline type: " + kind);
  }

 private:
  std::unique_ptr<gsl_spline, std::function<void(gsl_spline *)>> workspace_;
  Accelerator acc_;

  /// Initializes the interpolation object
  inline auto init(const Eigen::VectorXd &xa,
                   const Eigen::VectorXd &ya) noexcept -> void {
    acc_.reset();
    gsl_spline_init(workspace_.get(), xa.data(), ya.data(), xa.size());
  }
};

}  // namespace pyinterp::detail::gsl
