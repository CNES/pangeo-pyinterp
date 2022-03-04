// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <memory>
#include <utility>

#include "pyinterp/detail/math.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::math {

/// Set of coordinates used for interpolation
class CoordsXY {
 public:
  /// Default constructor
  CoordsXY() = delete;

  /// Creates a new instance
  CoordsXY(const Eigen::Index x_size, const Eigen::Index y_size)
      : x_(new Eigen::VectorXd), y_(new Eigen::VectorXd) {
    auto nx = x_size << 1U;
    auto ny = y_size << 1U;
    x_->resize(nx);
    y_->resize(ny);
  }

  /// Creates a new instance from existing coordinates
  CoordsXY(std::shared_ptr<Eigen::VectorXd> x,
           std::shared_ptr<Eigen::VectorXd> y)
      : x_(std::move(x)), y_(std::move(y)) {}

  /// Default destructor
  virtual ~CoordsXY() = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  CoordsXY(const CoordsXY &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  CoordsXY(CoordsXY &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const CoordsXY &rhs) -> CoordsXY & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(CoordsXY &&rhs) noexcept -> CoordsXY & = default;

  /// Get the half size of the window in abscissa.
  [[nodiscard]] inline auto nx() const noexcept -> Eigen::Index {
    return x_->size() >> 1;
  }

  /// Get the half size of the window in ordinate.
  [[nodiscard]] inline auto ny() const noexcept -> Eigen::Index {
    return y_->size() >> 1;
  }

  /// Get x-coordinates
  constexpr auto x() noexcept -> std::shared_ptr<Eigen::VectorXd> & {
    return x_;
  }

  /// Get x-coordinates
  [[nodiscard]] constexpr auto x() const noexcept
      -> const std::shared_ptr<Eigen::VectorXd> & {
    return x_;
  }

  /// Get y-coordinates
  constexpr auto y() noexcept -> std::shared_ptr<Eigen::VectorXd> & {
    return y_;
  }

  /// Get y-coordinates
  [[nodiscard]] constexpr auto y() const noexcept
      -> const std::shared_ptr<Eigen::VectorXd> & {
    return y_;
  }

  /// Get the ith x-axis.
  [[nodiscard]] inline auto x(const Eigen::Index ix) const -> double {
    return (*x_)(ix);
  }

  /// Get the ith y-axis.
  [[nodiscard]] inline auto y(const Eigen::Index jx) const -> double {
    return (*y_)(jx);
  }

  /// Set the ith x-axis.
  inline auto x(const Eigen::Index ix) -> double & { return (*x_)(ix); }

  /// Get the ith y-axis.
  inline auto y(const Eigen::Index jx) -> double & { return (*y_)(jx); }

  /// Normalizes the angle with respect to the first value of the X axis of this
  /// array.
  [[nodiscard]] inline auto normalize_angle(const double xi) const -> double {
    return math::normalize_angle(xi, (*x_)(0), 360.0);
  }

 private:
  std::shared_ptr<Eigen::VectorXd> x_{};
  std::shared_ptr<Eigen::VectorXd> y_{};
};

/// Set of coordinates/values used for interpolation
///  * q11 = (x1, y1)
///  * q12 = (x1, y2)
///  * .../...
///  * q1n = (x1, yn)
///  * q21 = (x2, y1)
///  * q22 = (x2, y2).
///  * .../...
///  * q2n = (x2, yn)
///  * .../...
///  * qnn = (xn, yn)
///
/// @code
/// Array2D({{x1, x2, ..., xn}, {y1, y2, ..., yn}},
///         {q11, q12, ..., q21, q22, ...., qnn})
/// @endcode
class Frame2D : public CoordsXY {
 public:
  /// Default constructor
  Frame2D() = delete;

  /// Creates a new Array
  Frame2D(const Eigen::Index x_size, const Eigen::Index y_size)
      : CoordsXY(x_size, y_size), q_(new Eigen::MatrixXd) {
    q_->resize(x()->size(), y()->size());
  }

  /// Creates a new Array from existing coordinates/values
  Frame2D(std::shared_ptr<Eigen::VectorXd> x,
          std::shared_ptr<Eigen::VectorXd> y,
          std::shared_ptr<Eigen::MatrixXd> q)
      : CoordsXY(std::move(x), std::move(y)), q_(std::move(q)) {}

  /// Default destructor
  ~Frame2D() override = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Frame2D(const Frame2D &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Frame2D(Frame2D &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Frame2D &rhs) -> Frame2D & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Frame2D &&rhs) noexcept -> Frame2D & = default;

  /// Get the values from the array for all x and y coordinates.
  constexpr auto q() noexcept -> std::shared_ptr<Eigen::MatrixXd> & {
    return q_;
  }

  /// Get the values from the array for all x and y coordinates.
  [[nodiscard]] constexpr auto q() const noexcept
      -> const std::shared_ptr<Eigen::MatrixXd> & {
    return q_;
  }

  /// Get the value at coordinate (ix, jx).
  [[nodiscard]] inline auto q(const Eigen::Index ix,
                              const Eigen::Index jx) const -> double {
    return (*q_)(ix, jx);
  }

  /// Get the value at coordinate (ix, jx).
  inline auto q(const Eigen::Index ix, const Eigen::Index jx) -> double & {
    return (*q_)(ix, jx);
  }

  /// Returns true if this instance does not contains at least one Not A Number
  /// (NaN).
  [[nodiscard]] inline auto is_valid() const -> bool { return !q_->hasNaN(); }

 private:
  std::shared_ptr<Eigen::MatrixXd> q_{};
};

/// Set of coordinates/values used for 3D-interpolation
///
/// @tparam Z-Axis type
template <typename T>
class Frame3D : public CoordsXY {
 public:
  /// Default constructor
  Frame3D() = delete;

  /// Creates a new instance
  Frame3D(const Eigen::Index x_size, const Eigen::Index y_size,
          const Eigen::Index z_size)
      : CoordsXY(x_size, y_size), z_() {
    auto nz = z_size << 1U;
    z_.resize(nz);
    q_.resize(nz);

    for (auto iz = 0U; iz < nz; ++iz) {
      q_(iz) = std::make_shared<Eigen::MatrixXd>(x()->size(), y()->size());
    }
  }

  /// Get the set of coordinates/values for the ith z-layer
  [[nodiscard]] auto frame_2d(const Eigen::Index iz) const -> Frame2D {
    return Frame2D(x(), y(), q_(iz));
  }

  /// Default destructor
  ~Frame3D() override = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Frame3D(const Frame3D &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Frame3D(Frame3D &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Frame3D &rhs) -> Frame3D & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Frame3D &&rhs) noexcept -> Frame3D & = default;

  /// Get the half size of the window in z.
  [[nodiscard]] inline auto nz() const noexcept -> Eigen::Index {
    return z_.size() >> 1;
  }

  /// Get z-coordinates
  constexpr auto z() noexcept -> Vector<T> & { return z_; }

  /// Get z-coordinates
  [[nodiscard]] constexpr auto z() const noexcept -> const Vector<T> & {
    return z_;
  }

  /// Get the ith z-axis.
  [[nodiscard]] inline auto z(const Eigen::Index ix) const -> T {
    return z_(ix);
  }

  /// Set the ith z-axis.
  inline auto z(const Eigen::Index ix) -> T & { return z_(ix); }

  /// Get the value at coordinate (ix, jx, kx).
  inline auto q(const Eigen::Index ix, const Eigen::Index jx,
                const Eigen::Index kx) -> double & {
    return (*q_(kx))(ix, jx);
  }

  /// Returns true if this instance does not contains at least one Not A Number
  /// (NaN).
  [[nodiscard]] inline auto is_valid() const -> bool {
    for (Eigen::Index kx = 0; kx < q_.size(); ++kx) {
      if ((*q_(kx)).hasNaN()) {
        return false;
      }
    }
    return true;
  }

 private:
  Vector<T> z_;
  Vector<std::shared_ptr<Eigen::MatrixXd>> q_;
};

/// Set of coordinates/values used for 4D-interpolation
///
/// @tparam Z-Axis type
template <typename T>
class Frame4D : public CoordsXY {
 public:
  /// Default constructor
  Frame4D() = delete;

  /// Creates a new instance
  Frame4D(const Eigen::Index x_size, const Eigen::Index y_size,
          const Eigen::Index z_size, const Eigen::Index u_size)
      : CoordsXY(x_size, y_size), z_() {
    auto nz = z_size << 1U;
    auto nu = u_size << 1U;
    z_.resize(nz);
    u_.resize(nu);
    q_.resize(nz, nu);

    for (auto iz = 0U; iz < nz; ++iz) {
      for (auto iu = 0U; iu < nu; ++iu) {
        q_(iz, iu) =
            std::make_shared<Eigen::MatrixXd>(x()->size(), y()->size());
      }
    }
  }

  /// Get the set of coordinates/values for the ith z-layer
  [[nodiscard]] auto frame_2d(const Eigen::Index iz,
                              const Eigen::Index iu) const -> Frame2D {
    return Frame2D(x(), y(), q_(iz, iu));
  }

  /// Default destructor
  ~Frame4D() override = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Frame4D(const Frame4D &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Frame4D(Frame4D &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Frame4D &rhs) -> Frame4D & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Frame4D &&rhs) noexcept -> Frame4D & = default;

  /// Get the half size of the window in z.
  [[nodiscard]] inline auto nz() const noexcept -> Eigen::Index {
    return z_.size() >> 1;
  }

  /// Get the half size of the window in u.
  [[nodiscard]] inline auto nu() const noexcept -> Eigen::Index {
    return u_.size() >> 1;
  }

  /// Get z-coordinates
  constexpr auto z() noexcept -> Vector<T> & { return z_; }

  /// Get u-coordinates
  constexpr auto u() noexcept -> Eigen::VectorXd & { return u_; }

  /// Get z-coordinates
  [[nodiscard]] inline auto z() const noexcept -> const Vector<T> & {
    return z_;
  }

  /// Get u-coordinates
  [[nodiscard]] inline auto u() const noexcept -> const Eigen::VectorXd & {
    return u_;
  }

  /// Get the ith z-axis.
  [[nodiscard]] inline auto z(const Eigen::Index ix) const -> T {
    return z_(ix);
  }

  /// Get the ith u-axis.
  [[nodiscard]] inline auto u(const Eigen::Index ix) const -> double {
    return u_(ix);
  }

  /// Set the ith z-axis.
  inline auto z(const Eigen::Index ix) -> T & { return z_(ix); }

  /// Set the ith u-axis.
  inline auto u(const Eigen::Index ix) -> double & { return u_(ix); }

  /// Get the value at coordinate (ix, jx, kx, lx).
  inline auto q(const Eigen::Index ix, const Eigen::Index jx,
                const Eigen::Index kx, const Eigen::Index lx) -> double & {
    return (*q_(kx, lx))(ix, jx);
  }

  /// Returns true if this instance does not contains at least one Not A Number
  /// (NaN).
  [[nodiscard]] inline auto is_valid() const -> bool {
    for (Eigen::Index kx = 0; kx < z_.size(); ++kx) {
      for (Eigen::Index lx = 0; lx < u_.size(); ++lx) {
        if ((*q_(kx, lx)).hasNaN()) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  Vector<T> z_;
  Eigen::VectorXd u_;
  Matrix<std::shared_ptr<Eigen::MatrixXd>> q_;
};

}  // namespace pyinterp::detail::math
