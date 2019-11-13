// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>
#include "pyinterp/axis.hpp"
#include "pyinterp/detail/broadcast.hpp"

namespace pyinterp {

/// Cartesian Grid 2D
template <typename T, ssize_t Dimension = 2>
class Grid2D {
 public:
  /// Default constructor
  Grid2D(std::shared_ptr<Axis> x, std::shared_ptr<Axis> y,
         pybind11::array_t<T> z)
      : x_(std::move(x)),
        y_(std::move(y)),
        array_(std::move(z)),
        ptr_(array_.template unchecked<Dimension>()) {
    check_shape(0, x_.get(), "x", "z", y_.get(), "y", "z");
  }

  /// Default constructor
  Grid2D() = default;

  /// Default destructor
  virtual ~Grid2D() = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Grid2D(const Grid2D& rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Grid2D(Grid2D&& rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Grid2D& rhs) -> Grid2D& = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Grid2D&& rhs) noexcept -> Grid2D& = default;

  /// Gets the X-Axis
  [[nodiscard]] inline auto x() const noexcept -> std::shared_ptr<Axis> {
    return x_;
  }

  /// Gets the Y-Axis
  [[nodiscard]] inline auto y() const noexcept -> std::shared_ptr<Axis> {
    return y_;
  }

  /// Gets values of the array to interpolate
  inline auto array() const noexcept -> const pybind11::array_t<T>& {
    return array_;
  }

  /// Gets the grid value for the coordinate pixel (ix, iy, ...).
  template <typename... Index>
  inline auto value(Index&&... index) const noexcept -> const T& {
    return ptr_(std::forward<Index>(index)...);
  }

  /// Throws an exception indicating that the value searched on the axis is
  /// outside the domain axis.
  ///
  /// @param axis Axis involved.
  /// @param value The value outside the axis domain.
  /// @param axis_label The name of the axis
  static void index_error(const Axis& axis, const double value,
                          const std::string& axis_label) {
    throw std::invalid_argument(std::to_string(value) +
                                " is out ouf bounds for axis " + axis_label +
                                " (" + static_cast<std::string>(axis) + ")");
  }

  /// Pickle support: get state of this instance
  [[nodiscard]] virtual auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(x_->getstate(), y_->getstate(), array_);
  }

  /// Pickle support: set state of this instance
  static auto setstate(const pybind11::tuple& tuple) -> Grid2D {
    if (tuple.size() != 3) {
      throw std::runtime_error("invalid state");
    }
    return Grid2D(std::make_shared<Axis>(
                      Axis(Axis::setstate(tuple[0].cast<pybind11::tuple>()))),
                  std::make_shared<Axis>(
                      Axis(Axis::setstate(tuple[1].cast<pybind11::tuple>()))),
                  tuple[2].cast<pybind11::array_t<T>>());
  }

 protected:
  std::shared_ptr<Axis> x_;
  std::shared_ptr<Axis> y_;
  pybind11::array_t<T> array_;
  pybind11::detail::unchecked_reference<T, Dimension> ptr_;

  /// End of the recursive call of the function "check_shape"
  void check_shape(const size_t idx) {}

  /// Checking the shape of the array for each defined axis.
  template <typename... Args>
  void check_shape(const size_t idx, const Axis* axis, const std::string& x,
                   const std::string& y, Args... args) {
    if (axis->size() != array_.shape(idx)) {
      throw std::invalid_argument(
          x + ", " + y + " could not be broadcast together with shape (" +
          std::to_string(axis->size()) + ", ) " +
          detail::ndarray_shape(array_));
    }
    check_shape(idx + 1, args...);
  }
};

/// Cartesian Grid 3D
template <typename T>
class Grid3D : public Grid2D<T, 3> {
 public:
  /// Default constructor
  Grid3D(const std::shared_ptr<Axis>& x, const std::shared_ptr<Axis>& y,
         std::shared_ptr<Axis> z, pybind11::array_t<T> u)
      : Grid2D<T, 3>(x, y, std::move(u)), z_(std::move(z)) {
    this->check_shape(2, z_.get(), "z", "u");
  }

  /// Gets the Y-Axis
  [[nodiscard]] inline auto z() const noexcept -> std::shared_ptr<Axis> {
    return z_;
  }

  /// Pickle support: get state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple final {
    return pybind11::make_tuple(this->x_->getstate(), this->y_->getstate(),
                                z_->getstate(), this->array_);
  }

  /// Pickle support: set state of this instance
  static auto setstate(const pybind11::tuple& tuple) -> Grid3D {
    if (tuple.size() != 4) {
      throw std::runtime_error("invalid state");
    }
    return Grid3D(std::make_shared<Axis>(
                      Axis::setstate(tuple[0].cast<pybind11::tuple>())),
                  std::make_shared<Axis>(
                      Axis::setstate(tuple[1].cast<pybind11::tuple>())),
                  std::make_shared<Axis>(
                      Axis::setstate(tuple[2].cast<pybind11::tuple>())),
                  tuple[3].cast<pybind11::array_t<T>>());
  }

 protected:
  std::shared_ptr<Axis> z_;
};

template <typename Type>
void implement_grid(pybind11::module& m, const std::string& suffix) {
  pybind11::class_<Grid2D<Type>>(m, ("Grid2D" + suffix).c_str(),
                                 "Cartesian Grid 2D")
      .def(pybind11::init<std::shared_ptr<Axis>, std::shared_ptr<Axis>,
                          pybind11::array_t<Type>>(),
           pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("z"),
           R"__doc__(
Default constructor

Args:
    x (pyinterp.core.Axis): X-Axis
    y (pyinterp.core.Axis): Y-Axis
    array (numpy.ndarray): Bivariate function
)__doc__")
      .def_property_readonly(
          "x", [](const Grid2D<Type>& self) { return self.x(); },
          R"__doc__(
Gets the X-Axis handled by this instance

Return:
    pyinterp.core.Axis: X-Axis
)__doc__")
      .def_property_readonly(
          "y", [](const Grid2D<Type>& self) { return self.y(); },
          R"__doc__(
Gets the Y-Axis handled by this instance

Return:
    pyinterp.core.Axis: Y-Axis
)__doc__")
      .def_property_readonly(
          "array", [](const Grid2D<Type>& self) { return self.array(); },
          R"__doc__(
Gets the values handled by this instance

Return:
    numpy.ndarray: values
)__doc__")
      .def(pybind11::pickle(
          [](const Grid2D<Type>& self) { return self.getstate(); },
          [](const pybind11::tuple& state) {
            return Grid2D<Type>::setstate(state);
          }));

  pybind11::class_<Grid3D<Type>>(m, ("Grid3D" + suffix).c_str(),
                                 "Cartesian Grid 3D")
      .def(pybind11::init<std::shared_ptr<Axis>, std::shared_ptr<Axis>,
                          std::shared_ptr<Axis>, pybind11::array_t<Type>>(),
           pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("z"),
           pybind11::arg("array"),
           R"__doc__(
Default constructor

Args:
    x (pyinterp.core.Axis): X-Axis
    y (pyinterp.core.Axis): Y-Axis
    z (pyinterp.core.Axis): Z-Axis
    array (numpy.ndarray): Trivariate function
)__doc__")
      .def_property_readonly(
          "x", [](const Grid3D<Type>& self) { return self.x(); },
          R"__doc__(
Gets the X-Axis handled by this instance

Return:
    pyinterp.core.Axis: X-Axis
)__doc__")
      .def_property_readonly(
          "y", [](const Grid3D<Type>& self) { return self.y(); },
          R"__doc__(
Gets the Y-Axis handled by this instance

Return:
    pyinterp.core.Axis: Y-Axis
)__doc__")
      .def_property_readonly(
          "z", [](const Grid3D<Type>& self) { return self.z(); },
          R"__doc__(
Gets the Z-Axis handled by this instance

Return:
    pyinterp.core.Axis: Z-Axis
)__doc__")
      .def_property_readonly(
          "array", [](const Grid3D<Type>& self) { return self.array(); },
          R"__doc__(
Gets the values handled by this instance

Return:
    numpy.ndarray: values
)__doc__")
      .def(pybind11::pickle(
          [](const Grid3D<Type>& self) { return self.getstate(); },
          [](const pybind11::tuple& state) {
            return Grid3D<Type>::setstate(state);
          }));
}

}  // namespace pyinterp
