// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>

#include <memory>
#include <string>
#include <utility>

#include "pyinterp/axis.hpp"
#include "pyinterp/detail/broadcast.hpp"

namespace pyinterp {

/// Cartesian Grid 2D
///
/// @tparam DataType Grid data type
/// @tparam Dimension Total number of dimensions handled by this instance.
template <typename DataType, pybind11::ssize_t Dimension = 2>
class Grid2D {
 public:
  /// Default constructor
  Grid2D(std::shared_ptr<Axis<double>> x, std::shared_ptr<Axis<double>> y,
         pybind11::array_t<DataType> array)
      : x_(std::move(x)),
        y_(std::move(y)),
        array_(std::move(array)),
        ptr_(array_.template unchecked<Dimension>()) {
    check_shape(0, x_.get(), "x", "array", y_.get(), "y", "array");
    if (y_->is_circle()) {
      throw std::invalid_argument("Y-axis cannot be a circle.");
    }
  }

  /// Default constructor
  Grid2D() = default;

  /// Default destructor
  virtual ~Grid2D() = default;

  /// Copy constructor
  ///
  /// @param rhs right value
  Grid2D(const Grid2D &rhs) = default;

  /// Move constructor
  ///
  /// @param rhs right value
  Grid2D(Grid2D &&rhs) noexcept = default;

  /// Copy assignment operator
  ///
  /// @param rhs right value
  auto operator=(const Grid2D &rhs) -> Grid2D & = default;

  /// Move assignment operator
  ///
  /// @param rhs right value
  auto operator=(Grid2D &&rhs) noexcept -> Grid2D & = default;

  /// Gets the X-Axis
  [[nodiscard]] inline auto x() const noexcept
      -> std::shared_ptr<Axis<double>> {
    return x_;
  }

  /// Gets the Y-Axis
  [[nodiscard]] inline auto y() const noexcept
      -> std::shared_ptr<Axis<double>> {
    return y_;
  }

  /// Gets values of the array to interpolate
  inline auto array() const noexcept -> const pybind11::array_t<DataType> & {
    return array_;
  }

  /// Gets the grid value for the coordinate pixel (ix, iy, ...).
  template <typename... Index>
  inline auto value(Index &&...index) const noexcept -> const DataType & {
    return ptr_(std::forward<Index>(index)...);
  }

  /// Throws an exception indicating that the value searched on the axis is
  /// outside the domain axis.
  ///
  /// @param axis Axis involved.
  /// @param value The value outside the axis domain.
  /// @param axis_label The name of the axis
  template <typename AxisType>
  static void index_error(const Axis<AxisType> &axis, const AxisType value,
                          const std::string &axis_label) {
    throw std::invalid_argument(
        axis.coordinate_repr(value) + " is out ouf bounds for axis " +
        axis_label + " [" + axis.coordinate_repr(axis.min_value()) + ", ..., " +
        axis.coordinate_repr(axis.max_value()) + "]");
  }

  /// Pickle support: get state of this instance
  [[nodiscard]] virtual auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(x_->getstate(), y_->getstate(), array_);
  }

  /// Pickle support: set state of this instance
  static auto setstate(const pybind11::tuple &tuple) -> Grid2D {
    if (tuple.size() != 3) {
      throw std::runtime_error("invalid state");
    }
    return Grid2D(std::make_shared<Axis<double>>(
                      Axis<double>::setstate(tuple[0].cast<pybind11::tuple>())),
                  std::make_shared<Axis<double>>(
                      Axis<double>::setstate(tuple[1].cast<pybind11::tuple>())),
                  tuple[2].cast<pybind11::array_t<DataType>>());
  }

 protected:
  std::shared_ptr<Axis<double>> x_;
  std::shared_ptr<Axis<double>> y_;
  pybind11::array_t<DataType> array_;
  pybind11::detail::unchecked_reference<DataType, Dimension> ptr_;

  /// End of the recursive call of the function "check_shape"
  void check_shape(const size_t idx) {}

  /// Checking the shape of the array for each defined axis.
  template <typename AxisType, typename... Args>
  void check_shape(const size_t idx, const Axis<AxisType> *axis,
                   const std::string &x, const std::string &y, Args... args) {
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
///
/// @tparam DataType Grid data type
/// @tparam AxisType Axis data type
/// @tparam Dimension Total number of dimensions handled by this instance.
template <typename DataType, typename AxisType, pybind11::ssize_t Dimension = 3>
class Grid3D : public Grid2D<DataType, Dimension> {
 public:
  /// Default constructor
  Grid3D(const std::shared_ptr<Axis<double>> &x,
         const std::shared_ptr<Axis<double>> &y,
         std::shared_ptr<Axis<AxisType>> z, pybind11::array_t<DataType> array)
      : Grid2D<DataType, Dimension>(x, y, std::move(array)), z_(std::move(z)) {
    this->check_shape(2, z_.get(), "z", "array");
    if (z_->is_circle()) {
      throw std::invalid_argument("Z-axis cannot be a circle.");
    }
  }

  /// Gets the Y-Axis
  [[nodiscard]] inline auto z() const noexcept
      -> std::shared_ptr<Axis<AxisType>> {
    return z_;
  }

  /// Pickle support: get state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple override {
    return pybind11::make_tuple(this->x_->getstate(), this->y_->getstate(),
                                z_->getstate(), this->array_);
  }

  /// Pickle support: set state of this instance
  static auto setstate(const pybind11::tuple &tuple) -> Grid3D {
    if (tuple.size() != 4) {
      throw std::runtime_error("invalid state");
    }
    return Grid3D(std::make_shared<Axis<double>>(
                      Axis<double>::setstate(tuple[0].cast<pybind11::tuple>())),
                  std::make_shared<Axis<double>>(
                      Axis<double>::setstate(tuple[1].cast<pybind11::tuple>())),
                  std::make_shared<Axis<AxisType>>(Axis<AxisType>::setstate(
                      tuple[2].cast<pybind11::tuple>())),
                  tuple[3].cast<pybind11::array_t<DataType>>());
  }

 protected:
  std::shared_ptr<Axis<AxisType>> z_;
};

/// Cartesian Grid 4D
///
/// @tparam DataType Grid data type
/// @tparam AxisType Axis data type
template <typename DataType, typename AxisType>
class Grid4D : public Grid3D<DataType, AxisType, 4> {
 public:
  /// Default constructor
  Grid4D(const std::shared_ptr<Axis<double>> &x,
         const std::shared_ptr<Axis<double>> &y,
         std::shared_ptr<Axis<AxisType>> z, std::shared_ptr<Axis<double>> u,
         pybind11::array_t<DataType> array)
      : Grid3D<DataType, AxisType, 4>(x, y, z, std::move(array)),
        u_(std::move(u)) {
    this->check_shape(3, u_.get(), "u", "array");
    if (u_->is_circle()) {
      throw std::invalid_argument("U-axis cannot be a circle.");
    }
  }

  /// Gets the U-Axis
  [[nodiscard]] inline auto u() const noexcept
      -> std::shared_ptr<Axis<double>> {
    return u_;
  }

  /// Pickle support: get state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple final {
    return pybind11::make_tuple(this->x_->getstate(), this->y_->getstate(),
                                this->z_->getstate(), u_->getstate(),
                                this->array_);
  }

  /// Pickle support: set state of this instance
  static auto setstate(const pybind11::tuple &tuple) -> Grid4D {
    if (tuple.size() != 5) {
      throw std::runtime_error("invalid state");
    }
    return Grid4D(std::make_shared<Axis<double>>(
                      Axis<double>::setstate(tuple[0].cast<pybind11::tuple>())),
                  std::make_shared<Axis<double>>(
                      Axis<double>::setstate(tuple[1].cast<pybind11::tuple>())),
                  std::make_shared<Axis<AxisType>>(Axis<AxisType>::setstate(
                      tuple[2].cast<pybind11::tuple>())),
                  std::make_shared<Axis<double>>(
                      Axis<double>::setstate(tuple[3].cast<pybind11::tuple>())),
                  tuple[4].cast<pybind11::array_t<DataType>>());
  }

 protected:
  std::shared_ptr<Axis<double>> u_;
};

/// Implementations of Cartesian grids with N dimensions.
///
/// @tparam DataType Grid data type
/// @tparam AxisType Axis data type
template <typename DataType, typename AxisType>
void implement_ndgrid(pybind11::module &m, const std::string &prefix,
                      const std::string &suffix) {
  std::string help = "Cartesian Grid 3D";
  if (prefix.length()) {
    help = prefix + " " + help;
  }
  pybind11::class_<Grid3D<DataType, AxisType>>(
      m, (prefix + "Grid3D" + suffix).c_str(),
      (prefix + "Grid3D" + suffix +
       "(self,"
       " x: pyinterp.core.Axis,"
       " y: pyinterp.core.Axis,"
       " z: pyinterp.core." +
       prefix +
       "Axis,"
       " array: numpy.ndarray)" +
       R"(

)" + help +
       R"(

Args:
    x: X-Axis
    y: Y-Axis
    z: Z-Axis
    array: Trivariate function values.
)")
          .c_str())
      .def(pybind11::init<
               std::shared_ptr<Axis<double>>, std::shared_ptr<Axis<double>>,
               std::shared_ptr<Axis<AxisType>>, pybind11::array_t<DataType>>(),
           pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("z"),
           pybind11::arg("array"))
      .def_property_readonly(
          "x", [](const Grid3D<DataType, AxisType> &self) { return self.x(); },
          R"__doc__(
Gets the X-Axis handled by this instance.

Returns:
    X-Axis.
)__doc__")
      .def_property_readonly(
          "y", [](const Grid3D<DataType, AxisType> &self) { return self.y(); },
          R"__doc__(
Gets the Y-Axis handled by this instance.

Returns:
    Y-Axis.
)__doc__")
      .def_property_readonly(
          "z", [](const Grid3D<DataType, AxisType> &self) { return self.z(); },
          R"__doc__(
Gets the Z-Axis handled by this instance.

Returns:
    Z-Axis.
)__doc__")
      .def_property_readonly(
          "array",
          [](const Grid3D<DataType, AxisType> &self) { return self.array(); },
          R"__doc__(
Gets the values handled by this instance.

Returns:
    Values.
)__doc__")
      .def(pybind11::pickle(
          [](const Grid3D<DataType, AxisType> &self) {
            return self.getstate();
          },
          [](const pybind11::tuple &state) {
            return Grid3D<DataType, AxisType>::setstate(state);
          }));

  help = "Cartesian Grid 4D";
  if (prefix.length()) {
    help = prefix + " " + help;
  }
  pybind11::class_<Grid4D<DataType, AxisType>>(
      m, (prefix + "Grid4D" + suffix).c_str(),
      (prefix + "Grid4D" + suffix +
       "(self,"
       " x: pyinterp.core.Axis,"
       " y: pyinterp.core.Axis,"
       " z: pyinterp.core.Axis,"
       " u: pyinterp.core." +
       prefix +
       "Axis,"
       " array: numpy.ndarray)" +
       R"(

)" + help +
       R"(

Args:
    x: X-Axis
    y: Y-Axis
    z: Z-Axis
    u: U-Axis
    array: Quadrivariate function
)")
          .c_str())
      .def(pybind11::init<
               std::shared_ptr<Axis<double>>, std::shared_ptr<Axis<double>>,
               std::shared_ptr<Axis<AxisType>>, std::shared_ptr<Axis<double>>,
               pybind11::array_t<DataType>>(),
           pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("z"),
           pybind11::arg("u"), pybind11::arg("array"))
      .def_property_readonly(
          "x", [](const Grid4D<DataType, AxisType> &self) { return self.x(); },
          R"__doc__(
Gets the X-Axis handled by this instance.

Returns:
    X-Axis.
)__doc__")
      .def_property_readonly(
          "y", [](const Grid4D<DataType, AxisType> &self) { return self.y(); },
          R"__doc__(
Gets the Y-Axis handled by this instance.

Returns:
    Y-Axis.
)__doc__")
      .def_property_readonly(
          "z", [](const Grid4D<DataType, AxisType> &self) { return self.z(); },
          R"__doc__(
Gets the Z-Axis handled by this instance.

Returns:
    Z-Axis.
)__doc__")
      .def_property_readonly(
          "u", [](const Grid4D<DataType, AxisType> &self) { return self.u(); },
          R"__doc__(
Gets the U-Axis handled by this instance.

Returns:
    U-Axis.
)__doc__")
      .def_property_readonly(
          "array",
          [](const Grid4D<DataType, AxisType> &self) { return self.array(); },
          R"__doc__(
Gets the values handled by this instance.

Returns:
    Values.
)__doc__")
      .def(pybind11::pickle(
          [](const Grid4D<DataType, AxisType> &self) {
            return self.getstate();
          },
          [](const pybind11::tuple &state) {
            return Grid4D<DataType, AxisType>::setstate(state);
          }));
}

/// Implementations of Cartesian grids.
///
/// @tparam DataType Grid data type
template <typename DataType>
void implement_grid(pybind11::module &m, const std::string &suffix) {
  pybind11::class_<Grid2D<DataType>>(m, ("Grid2D" + suffix).c_str(),
                                     ("Grid2D" + suffix +
                                      "(self,"
                                      " x: pyinterp.core.Axis,"
                                      " y: pyinterp.core.Axis,"
                                      " array: numpy.ndarray)" +
                                      R"__doc__(
Cartesian Grid 2D.

Args:
    x: X-Axis.
    y: Y-Axis.
    array: Bivariate function.
)__doc__")
                                         .c_str())
      .def(pybind11::init<std::shared_ptr<Axis<double>>,
                          std::shared_ptr<Axis<double>>,
                          pybind11::array_t<DataType>>(),
           pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("array"))
      .def_property_readonly(
          "x", [](const Grid2D<DataType> &self) { return self.x(); },
          R"__doc__(
Gets the X-Axis handled by this instance.

Returns:
    X-Axis.
)__doc__")
      .def_property_readonly(
          "y", [](const Grid2D<DataType> &self) { return self.y(); },
          R"__doc__(
Gets the Y-Axis handled by this instance.

Returns:
    Y-Axis.
)__doc__")
      .def_property_readonly(
          "array", [](const Grid2D<DataType> &self) { return self.array(); },
          R"__doc__(
Gets the values handled by this instance.

Returns:
    Values.
)__doc__")
      .def(pybind11::pickle(
          [](const Grid2D<DataType> &self) { return self.getstate(); },
          [](const pybind11::tuple &state) {
            return Grid2D<DataType>::setstate(state);
          }));

  implement_ndgrid<DataType, double>(m, "", suffix);
  implement_ndgrid<DataType, int64_t>(m, "Temporal", suffix);
}

}  // namespace pyinterp
