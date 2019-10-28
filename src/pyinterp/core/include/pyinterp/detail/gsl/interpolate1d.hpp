// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <gsl/gsl_interp.h>
#include <Eigen/Core>
#include <functional>
#include <memory>
#include "pyinterp/detail/broadcast.hpp"
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
  Interpolate1D(const size_t size, const gsl_interp_type* type, Accelerator acc)
      : workspace_(
            std::unique_ptr<gsl_interp, std::function<void(gsl_interp*)>>(
                gsl_interp_alloc(type, size),
                [](gsl_interp* ptr) { gsl_interp_free(ptr); })),
        acc_(std::move(acc)) {}

  /// Returns the name of the interpolation type used
  [[nodiscard]] inline auto name() const noexcept -> std::string {
    return gsl_interp_name(workspace_.get());
  }

  /// Return the minimum number of points required by the interpolation
  [[nodiscard]] inline auto min_size() const noexcept -> size_t {
    return gsl_interp_min_size(workspace_.get());
  }

  /// Return the interpolated value of y for a given point x
  inline auto interpolate(const Eigen::VectorXd& xa, const Eigen::VectorXd& ya,
                          const double x) -> double {
    init(xa, ya);
    return gsl_interp_eval(workspace_.get(), xa.data(), ya.data(), x, acc_);
  }

  /// Return the derivative d of an interpolated function for a given point x
  inline auto derivative(const Eigen::VectorXd& xa, const Eigen::VectorXd& ya,
                         const double x) -> double {
    init(xa, ya);
    return gsl_interp_eval_deriv(workspace_.get(), xa.data(), ya.data(), x,
                                 acc_);
  }

  /// Return the second derivative d of an interpolated function for a given
  /// point x
  inline auto second_derivative(const Eigen::VectorXd& xa,
                                const Eigen::VectorXd& ya, const double x)
      -> double {
    init(xa, ya);
    return gsl_interp_eval_deriv2(workspace_.get(), xa.data(), ya.data(), x,
                                  acc_);
  }

  /// Return the numerical integral result of an interpolated function over the
  /// range [a, b],
  inline auto integral(const Eigen::VectorXd& xa, const Eigen::VectorXd& ya,
                       const double a, const double b) -> double {
    init(xa, ya);
    return gsl_interp_eval_integ(workspace_.get(), xa.data(), ya.data(), a, b,
                                 acc_);
  }

 private:
  std::unique_ptr<gsl_interp, std::function<void(gsl_interp*)>> workspace_;
  Accelerator acc_;

  /// Initializes the interpolation object
  void init(const Eigen::VectorXd& xa, const Eigen::VectorXd& ya) {
    acc_.reset();
    gsl_interp_init(workspace_.get(), xa.data(), ya.data(), xa.size());
  }
};

}  // namespace pyinterp::detail::gsl
