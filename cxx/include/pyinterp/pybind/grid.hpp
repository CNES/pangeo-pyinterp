// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <sys/types.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "pyinterp/format_byte.hpp"
#include "pyinterp/pybind/axis.hpp"
#include "pyinterp/pybind/temporal_axis.hpp"

namespace pyinterp::pybind {
namespace detail {

/// Dtype name traits.
/// @tparam T Data type.
template <typename T>
struct dtype_name_traits;

/// @brief Specializations of dtype_name_traits for int8_t.
template <>
struct dtype_name_traits<int8_t> {
  static constexpr const char* value = "int8";
};

/// @brief Specializations of dtype_name_traits for uint8_t.
template <>
struct dtype_name_traits<uint8_t> {
  static constexpr const char* value = "uint8";
};

/// @brief Specializations of dtype_name_traits for int16_t.
template <>
struct dtype_name_traits<int16_t> {
  static constexpr const char* value = "int16";
};

/// @brief Specializations of dtype_name_traits for uint16_t.
template <>
struct dtype_name_traits<uint16_t> {
  static constexpr const char* value = "uint16";
};

/// @brief Specializations of dtype_name_traits for int32_t.
template <>
struct dtype_name_traits<int32_t> {
  static constexpr const char* value = "int32";
};

/// @brief Specializations of dtype_name_traits for uint32_t.
template <>
struct dtype_name_traits<uint32_t> {
  static constexpr const char* value = "uint32";
};

/// @brief Specializations of dtype_name_traits for int64_t.
template <>
struct dtype_name_traits<int64_t> {
  static constexpr const char* value = "int64";
};

/// @brief Specializations of dtype_name_traits for uint64_t.
template <>
struct dtype_name_traits<uint64_t> {
  static constexpr const char* value = "uint64";
};

/// @brief Specializations of dtype_name_traits for float.
template <>
struct dtype_name_traits<float> {
  static constexpr const char* value = "float32";
};

/// @brief Specializations of dtype_name_traits for double.
template <>
struct dtype_name_traits<double> {
  static constexpr const char* value = "float64";
};

/// Get the dtype name for a given data type.
/// @tparam T Data type.
/// @return Dtype name as a C string.
template <typename T>
[[nodiscard]] constexpr auto dtype_name() -> const char* {
  if constexpr (requires { dtype_name_traits<T>::value; }) {
    return dtype_name_traits<T>::value;
  } else {
    std::unreachable();
  }
}

/// Get a string representation of the shape of a NumPy array.
/// @tparam NDIMS Number of dimensions.
/// @tparam NDArray NumPy array type.
/// @param array NumPy array.
/// @return String representation of the shape.
template <size_t NDIMS, typename NDArray>
[[nodiscard]] inline auto array_shape_str(const NDArray& array) -> std::string {
  std::string shape_str = "(";
  for (size_t i = 0; i < NDIMS; ++i) {
    if (i > 0) {
      shape_str += ", ";
    }
    shape_str += std::to_string(array.shape(i));
  }
  shape_str += ")";
  return shape_str;
}

/// Trait to map math axis types to their nanobind wrappers.
template <typename MathAxisT>
struct axis_pybind_wrapper;

/// Specialization for math::Axis
/// @tparam T Value type of the axis
template <typename T>
struct axis_pybind_wrapper<math::Axis<T>> {
  using type = Axis<T>;
};

/// Specialization for math::TemporalAxis
template <>
struct axis_pybind_wrapper<math::TemporalAxis> {
  using type = TemporalAxis;
};

/// Helper alias to get the nanobind wrapper type for a math axis type.
/// @tparam MathAxisT Math axis type.
template <typename MathAxisT>
using axis_pybind_wrapper_t = typename axis_pybind_wrapper<MathAxisT>::type;

/// Helper to build tuple of pybind axis types from math axis types.
/// @tparam MathAxes Math axis types.
template <typename... MathAxes>
struct pybind_axes_tuple_builder {
  using type = std::tuple<axis_pybind_wrapper_t<MathAxes>...>;
};

/// Helper alias to get the pybind axes tuple type.
/// @tparam MathAxes Math axis types.
template <typename... MathAxes>
using pybind_axes_tuple_t =
    typename pybind_axes_tuple_builder<MathAxes...>::type;

}  // namespace detail

/// Concept for a valid axis.
template <typename T>
concept AxisTypeConcept = requires(const T& ax, typename T::value_type val) {
  typename T::value_type;
  { ax.size() } -> std::convertible_to<int64_t>;
  { ax.min_value() } -> std::same_as<typename T::value_type>;
  { ax.max_value() } -> std::same_as<typename T::value_type>;
  {
    ax.find_indexes(val)
  } -> std::same_as<std::optional<std::pair<int64_t, int64_t>>>;
  { ax.is_periodic() } -> std::same_as<bool>;
  { ax.coordinate_repr(val) } -> std::convertible_to<std::string>;
};

/// Generic N-dimensional Cartesian grid.
/// @tparam DataType Type of data stored in the grid.
/// @tparam MathAxes Axis types (math::Axis<double>, math::TemporalAxis, etc.).
template <typename DataType, AxisTypeConcept... MathAxes>
  requires(sizeof...(MathAxes) >= 1 && sizeof...(MathAxes) <= 4)
class Grid {
 public:
  /// The data type of the grid values.
  using data_type = DataType;

  /// Number of dimensions.
  static constexpr size_t kNDim = sizeof...(MathAxes);

  /// Tuple of axes.
  using math_axes_tuple_t = std::tuple<MathAxes...>;

  /// Math axis type at index I.
  /// @tparam I Index of the axis.
  template <size_t I>
  using math_axis_t = std::tuple_element_t<I, math_axes_tuple_t>;

  /// Nanobind axis type at index I (for Python-facing methods).
  /// @tparam I Index of the axis.
  template <size_t I>
  using pybind_axis_t = detail::axis_pybind_wrapper_t<math_axis_t<I>>;

  /// Tuple of pybind axes (stored internally).
  using pybind_axes_tuple_t = detail::pybind_axes_tuple_t<MathAxes...>;

  /// Extract the value type of the math axis at index I.
  /// @tparam I Index of the axis.
  template <size_t I>
  using math_axis_value_t = typename math_axis_t<I>::value_type;

  /// N-dimensional array type.
  using array_t = nanobind::ndarray<nanobind::numpy, DataType,
                                    nanobind::ndim<kNDim>, nanobind::c_contig>;

  /// N-dimensional array view type.
  using view_t = nanobind::ndarray_view<DataType, kNDim, 'C'>;

  /// Constructor.
  /// @param[in] axes Axes of the grid.
  /// @param[in] array N-dimensional data array.
  explicit Grid(detail::axis_pybind_wrapper_t<MathAxes>... axes, array_t array)
      : pybind_axes_{std::move(axes)...},
        array_{std::move(array)},
        ptr_{array_.template view<DataType>()} {
    validate_construction();
  }

  /// Constructor taking math axis types.
  /// @param[in] axes Axes of the grid.
  /// @param[in] array N-dimensional data array.
  explicit Grid(MathAxes... axes, array_t array)
      : Grid(detail::axis_pybind_wrapper_t<MathAxes>(std::move(axes))...,
             std::move(array)) {}

  /// Default constructor.
  Grid() = default;

  /// Destructor.
  ~Grid() = default;

  /// Copy/move semantics.
  Grid(const Grid&) = default;
  Grid(Grid&&) noexcept = default;
  auto operator=(const Grid&) -> Grid& = default;
  auto operator=(Grid&&) noexcept -> Grid& = default;

  /// @brief Get the data type as a string.
  /// @return Data type string.
  [[nodiscard]] auto dtype_str() const -> std::string_view {
    return detail::dtype_name<DataType>();
  }

  /// @brief Get the number of dimensions.
  /// @return Number of dimensions.
  [[nodiscard]] auto ndim() const -> size_t { return kNDim; }

  /// @brief Check if this grid has a temporal axis.
  /// @return True if the grid has a temporal axis, false otherwise.
  [[nodiscard]] auto has_temporal_axis() const -> bool {
    return kHasTemporalAxis;
  }

  /// @brief Get the index of the temporal axis.
  /// @return Index of the temporal axis, or -1 if no temporal axis.
  [[nodiscard]] auto temporal_axis_index() const -> int {
    return temporal_axis_index_impl(std::index_sequence_for<MathAxes...>{});
  }

  /// @brief Get the size of a specific axis.
  /// @param dim Dimension index.
  /// @return Size of the axis.
  [[nodiscard]] auto axis_size(size_t dim) const -> int64_t {
    return axis_size_impl(dim, std::index_sequence_for<MathAxes...>{});
  }

  /// @brief Check if a specific axis is periodic.
  /// @param dim Dimension index.
  /// @return True if the axis is periodic, false otherwise.
  [[nodiscard]] auto axis_is_periodic(size_t dim) const -> bool {
    return axis_is_periodic_impl(dim, std::index_sequence_for<MathAxes...>{});
  }

  /// @brief Get the shape of the grid as a vector.
  /// @return Vector containing the shape of each dimension.
  [[nodiscard]] auto shape() const -> std::vector<size_t> {
    return shape_impl(std::index_sequence_for<MathAxes...>{});
  }

  /// @brief Get the pybind axis object at a specific dimension.
  /// @param dim Dimension index.
  /// @return The axis as a nanobind::object.
  [[nodiscard]] auto pybind_axis_object(size_t dim) const -> nanobind::object {
    return pybind_axis_object_impl(dim, std::index_sequence_for<MathAxes...>{});
  }

  /// @brief Get the underlying data array as a nanobind::object.
  /// @return The data array.
  [[nodiscard]] auto array_object() const -> nanobind::object {
    return nanobind::cast(array_);
  }

  /// @brief Get a string representation of the grid.
  /// @return String representation.
  [[nodiscard]] auto repr() const -> std::string {
    return static_cast<std::string>(*this);
  }

  /// Get axis at index I at compile time.
  /// @tparam I Index of the axis.
  /// @return Reference to the axis.
  template <size_t I>
    requires(I < kNDim)
  [[nodiscard]] constexpr auto axis() const noexcept -> const math_axis_t<I>& {
    // Return the base class of the pybind axis (implicit conversion)
    return static_cast<const math_axis_t<I>&>(std::get<I>(pybind_axes_));
  }

  /// Get nanobind axis at index I at compile time.
  /// @tparam I Index of the axis.
  /// @return Reference to the stored pybind axis.
  template <size_t I>
    requires(I < kNDim)
  [[nodiscard]] constexpr auto pybind_axis() const noexcept
      -> const pybind_axis_t<I>& {
    return std::get<I>(pybind_axes_);
  }

  /// Get the underlying data array.
  /// @return Reference to the data array.
  [[nodiscard]] constexpr auto array() const noexcept -> const array_t& {
    return array_;
  }

  /// Get the grid value at specified indices.
  /// @tparam Index Types of the indices.
  /// @param[in] indices Indices along each axis.
  /// @return Reference to the data value.
  template <typename... Index>
    requires(sizeof...(Index) == kNDim)
  [[nodiscard]] constexpr auto value(Index&&... indices) const noexcept
      -> const DataType& {
    return ptr_(std::forward<Index>(indices)...);
  }

  /// Check if a value is within bounds for axis I.
  /// @tparam I Index of the axis.
  /// @param[in] coordinate Value to check.
  /// @return True if the value is within bounds, false otherwise.
  template <size_t I>
  [[nodiscard]] constexpr auto is_within_bounds(
      const math_axis_value_t<I>& coordinate) const noexcept -> bool {
    const auto& ax = axis<I>();
    return coordinate >= ax.min_value() && coordinate <= ax.max_value();
  }

  /// Construct an error description for out-of-bounds access on axis I.
  /// @tparam I Index of the axis.
  /// @param[in] coordinate Value that is out of bounds.
  /// @return Error description string.
  template <size_t I>
  [[nodiscard]] auto construct_bounds_error_description(
      const math_axis_value_t<I>& coordinate) const -> std::string {
    const auto& ax = axis<I>();
    constexpr std::array<std::string_view, 4> labels{"x", "y", "z", "u"};
    return std::format("{} is out of bounds for axis {} [{}, ..., {}]",
                       ax.coordinate_repr(coordinate), labels[I],
                       ax.coordinate_repr(ax.min_value()),
                       ax.coordinate_repr(ax.max_value()));
  }

  /// Throw an out-of-bounds error for axis I.
  /// @tparam I Index of the axis.
  /// @param[in] coordinate Value that is out of bounds.
  template <size_t I>
  [[noreturn]] auto throw_bounds_error(
      const math_axis_value_t<I>& coordinate) const -> void {
    throw std::invalid_argument(
        construct_bounds_error_description<I>(coordinate));
  }

  /// Find the indexes that surround a given coordinate along axis I.
  /// @tparam I Index of the axis.
  /// @param[in] coordinate Coordinate value.
  /// @param[in] bounds_error Whether to raise an error if out of bounds.
  /// @return Pair of surrounding indexes, or `std::nullopt` if out of bounds.
  template <size_t I>
  [[nodiscard]] constexpr auto find_indexes(
      const math_axis_value_t<I>& coordinate, const bool bounds_error) const
      -> std::optional<std::pair<size_t, size_t>> {
    const auto& ax = axis<I>();
    auto indexes = ax.find_indexes(coordinate);
    if (!indexes.has_value() && bounds_error) {
      this->template throw_bounds_error<I>(coordinate);
    }
    return indexes;
  }

  /// Get the state for pickling.
  /// @return Tuple representing the state.
  [[nodiscard]] auto getstate() const -> nanobind::tuple {
    return getstate_impl(std::index_sequence_for<MathAxes...>{});
  }

  /// Set the state from unpickling.
  /// @param[in] state Tuple representing the state.
  /// @return Reconstructed `Grid` object.
  [[nodiscard]] static auto setstate(const nanobind::tuple& state) -> Grid {
    if (state.size() != kNDim + 1) {
      throw std::runtime_error(
          std::format("invalid state: expected {} elements, got {}", kNDim + 1,
                      state.size()));
    }
    return setstate_impl(state, std::index_sequence_for<MathAxes...>{});
  }

  /// Convert the grid to a string representation.
  /// @return String representation of the grid showing dimensions, shape,
  /// dtype, and memory size.
  [[nodiscard]] explicit operator std::string() const {
    constexpr std::array<std::string_view, 4> dim_names{"1D", "2D", "3D", "4D"};
    std::string_view prefix = has_temporal_axis() ? "Temporal" : "";

    return std::format("{}Grid{}(shape={}, dtype={}, nbytes={})", prefix,
                       dim_names[kNDim - 1],
                       detail::array_shape_str<kNDim>(array_), dtype_str(),
                       format_bytes(array_.nbytes()));
  }

  /// @brief Check if this grid has a temporal axis.
  /// @return True if the grid has a temporal axis, false otherwise.
  static constexpr bool kHasTemporalAxis =
      (std::is_same_v<MathAxes, math::TemporalAxis> || ...);

 protected:
  pybind_axes_tuple_t pybind_axes_;
  array_t array_;
  view_t ptr_;

 private:
  /// Validate grid construction.
  auto validate_construction() -> void {
    validate_axes(std::index_sequence_for<MathAxes...>{});
    validate_shapes(std::index_sequence_for<MathAxes...>{});
  }

  /// Validate that circular axes are only allowed on the first axis.
  template <size_t... Is>
  auto validate_axes(std::index_sequence<Is...>) -> void {
    (
        [&] {
          if constexpr (Is > 0) {
            const auto& ax = axis<Is>();
            if (ax.is_periodic()) {
              constexpr std::array<std::string_view, 4> labels{"x", "y", "z",
                                                               "u"};
              throw std::invalid_argument(
                  std::format("{}-axis cannot be a circle", labels[Is]));
            }
          }
        }(),
        ...);
  }

  /// Validate that axis sizes match array shapes.
  template <size_t... Is>
  auto validate_shapes(std::index_sequence<Is...>) -> void {
    (
        [&] {
          const auto& ax = axis<Is>();
          if (std::cmp_not_equal(ax.size(), array_.shape(Is))) {
            constexpr std::array<std::string_view, 4> labels{"x", "y", "z",
                                                             "u"};
            throw std::invalid_argument(std::format(
                "{} axis size ({}) doesn't match array shape[{}] ({})",
                labels[Is], ax.size(), Is, array_.shape(Is)));
          }
        }(),
        ...);
  }

  /// Serialize the grid state for pickling.
  template <size_t... Is>
  [[nodiscard]] auto getstate_impl(std::index_sequence<Is...>) const
      -> nanobind::tuple {
    return nanobind::make_tuple(pybind_axis<Is>().getstate()..., array_);
  }

  /// Deserialize the grid state from unpickling.
  template <size_t... Is>
  [[nodiscard]] static auto setstate_impl(const nanobind::tuple& state,
                                          std::index_sequence<Is...>) -> Grid {
    return Grid(static_cast<math_axis_t<Is>>(pybind_axis_t<Is>::setstate(
                    nanobind::cast<nanobind::tuple>(state[Is])))...,
                nanobind::cast<array_t>(state[kNDim]));
  }

  /// @brief Find the index of the temporal axis.
  template <size_t... Is>
  [[nodiscard]] static constexpr auto temporal_axis_index_impl(
      std::index_sequence<Is...>) -> int {
    int result = -1;
    (void)((std::is_same_v<math_axis_t<Is>, math::TemporalAxis>
                ? (result = static_cast<int>(Is), true)
                : false) ||
           ...);
    return result;
  }

  /// @brief Get the size of a specific axis at runtime.
  template <size_t... Is>
  [[nodiscard]] auto axis_size_impl(size_t dim,
                                    std::index_sequence<Is...>) const
      -> int64_t {
    int64_t result = 0;
    (void)((dim == Is ? (result = static_cast<int64_t>(axis<Is>().size()), true)
                      : false) ||
           ...);
    return result;
  }

  /// @brief Check if a specific axis is periodic at runtime.
  template <size_t... Is>
  [[nodiscard]] auto axis_is_periodic_impl(size_t dim,
                                           std::index_sequence<Is...>) const
      -> bool {
    bool result = false;
    (void)((dim == Is ? (result = axis<Is>().is_periodic(), true) : false) ||
           ...);
    return result;
  }

  /// @brief Get the shape of the grid.
  template <size_t... Is>
  [[nodiscard]] auto shape_impl(std::index_sequence<Is...>) const
      -> std::vector<size_t> {
    return {static_cast<size_t>(axis<Is>().size())...};
  }

  /// @brief Get the pybind axis object at a specific dimension at runtime.
  template <size_t... Is>
  [[nodiscard]] auto pybind_axis_object_impl(size_t dim,
                                             std::index_sequence<Is...>) const
      -> nanobind::object {
    nanobind::object result = nanobind::none();
    (void)((dim == Is ? (result = nanobind::cast(pybind_axis<Is>()), true)
                      : false) ||
           ...);
    return result;
  }
};

/// Spatial axis alias for clarity.
template <typename T = double>
using MathSpatialAxis = math::Axis<T>;

/// Temporal axis alias for clarity.
using MathTemporalAxis = math::TemporalAxis;

/// One-dimensional grid alias.
/// @tparam DataType Type of data stored in the grid.
template <typename DataType>
using Grid1D = Grid<DataType, MathSpatialAxis<>>;

/// Two-dimensional grid alias.
/// @tparam DataType Type of data stored in the grid.
template <typename DataType>
using Grid2D = Grid<DataType, MathSpatialAxis<>, MathSpatialAxis<>>;

/// Three-dimensional grid alias.
/// @tparam DataType Type of data stored in the grid.
template <typename DataType>
using Grid3D =
    Grid<DataType, MathSpatialAxis<>, MathSpatialAxis<>, MathSpatialAxis<>>;

/// Four-dimensional grid alias.
/// @tparam DataType Type of data stored in the grid.
template <typename DataType>
using Grid4D = Grid<DataType, MathSpatialAxis<>, MathSpatialAxis<>,
                    MathSpatialAxis<>, MathSpatialAxis<>>;

/// Temporal three-dimensional grid alias.
/// @tparam DataType Type of data stored in the grid.
template <typename DataType>
using TemporalGrid3D =
    Grid<DataType, MathSpatialAxis<>, MathSpatialAxis<>, MathTemporalAxis>;

/// Temporal four-dimensional grid alias.
/// @tparam DataType Type of data stored in the grid.
template <typename DataType>
using TemporalGrid4D = Grid<DataType, MathSpatialAxis<>, MathSpatialAxis<>,
                            MathTemporalAxis, MathSpatialAxis<>>;

namespace detail {

/// @brief Helper to generate all grid types for a given data type.
/// @tparam T Data type.
template <typename T>
using GridsForType = std::variant<Grid1D<T>, Grid2D<T>, Grid3D<T>, Grid4D<T>,
                                  TemporalGrid3D<T>, TemporalGrid4D<T>>;

/// @brief List of supported data types for grids.
using SupportedDataTypes =
    std::tuple<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t,
               uint64_t, float, double>;

/// @brief Helper to concatenate variants.
template <typename... Variants>
struct variant_concat;

template <typename... Ts>
struct variant_concat<std::variant<Ts...>> {
  using type = std::variant<Ts...>;
};

template <typename... Ts, typename... Us, typename... Rest>
struct variant_concat<std::variant<Ts...>, std::variant<Us...>, Rest...> {
  using type =
      typename variant_concat<std::variant<Ts..., Us...>, Rest...>::type;
};

template <typename... Variants>
using variant_concat_t = typename variant_concat<Variants...>::type;

/// @brief Helper to generate GridVariant from tuple of data types.
template <typename DataTypeTuple,
          typename = std::make_index_sequence<std::tuple_size_v<DataTypeTuple>>>
struct GridVariantBuilder;

template <typename... DataTypes, size_t... Is>
struct GridVariantBuilder<std::tuple<DataTypes...>,
                          std::index_sequence<Is...>> {
  using type = variant_concat_t<GridsForType<DataTypes>...>;
};

}  // namespace detail

/// @brief Sum type holding all possible grid types.
/// Contains 60 types: 6 grid shapes × 10 data types.
using GridVariant =
    typename detail::GridVariantBuilder<detail::SupportedDataTypes>::type;

/// @brief Type-erased grid holder using std::variant internally.
///
/// This class wraps a GridVariant and provides a uniform interface for
/// accessing grid properties regardless of the underlying type.
class GridHolder {
 public:
  /// @brief Default constructor (required for pickle).
  GridHolder() = default;

  /// @brief Construct from any grid type.
  /// @tparam GridType The concrete grid type.
  /// @param[in,out] grid The grid to store.
  template <typename GridType>
    requires(!std::is_same_v<std::decay_t<GridType>, GridHolder>)
  explicit GridHolder(GridType&& grid) : value_(std::forward<GridType>(grid)) {}

  /// @brief Get the data type as a string.
  /// @return Data type string.
  [[nodiscard]] auto dtype_str() const -> std::string_view {
    return std::visit([](const auto& g) { return g.dtype_str(); }, value_);
  }

  /// @brief Get the number of dimensions.
  /// @return Number of dimensions.
  [[nodiscard]] auto ndim() const -> size_t {
    return std::visit([](const auto& g) { return g.ndim(); }, value_);
  }

  /// @brief Check if this grid has a temporal axis.
  /// @return True if the grid has a temporal axis, false otherwise.
  [[nodiscard]] auto has_temporal_axis() const -> bool {
    return std::visit([](const auto& g) { return g.has_temporal_axis(); },
                      value_);
  }

  /// @brief Get the index of the temporal axis.
  /// @return Index of the temporal axis, or -1 if no temporal axis.
  [[nodiscard]] auto temporal_axis_index() const -> int {
    return std::visit([](const auto& g) { return g.temporal_axis_index(); },
                      value_);
  }

  /// @brief Get the size of a specific axis.
  /// @param[in] dim Dimension index.
  /// @return Size of the axis.
  [[nodiscard]] auto axis_size(size_t dim) const -> int64_t {
    return std::visit([dim](const auto& g) { return g.axis_size(dim); },
                      value_);
  }

  /// @brief Check if a specific axis is periodic.
  /// @param[in] dim Dimension index.
  /// @return True if the axis is periodic, false otherwise.
  [[nodiscard]] auto axis_is_periodic(size_t dim) const -> bool {
    return std::visit([dim](const auto& g) { return g.axis_is_periodic(dim); },
                      value_);
  }

  /// @brief Get the shape of the grid as a vector.
  /// @return Vector containing the shape of each dimension.
  [[nodiscard]] auto shape() const -> std::vector<size_t> {
    return std::visit([](const auto& g) { return g.shape(); }, value_);
  }

  /// @brief Get the pybind axis object at a specific dimension.
  /// @param[in] dim Dimension index.
  /// @return The axis as a nanobind::object.
  [[nodiscard]] auto pybind_axis_object(size_t dim) const -> nanobind::object {
    return std::visit(
        [dim](const auto& g) { return g.pybind_axis_object(dim); }, value_);
  }

  /// @brief Get the underlying data array as a nanobind::object.
  /// @return The data array.
  [[nodiscard]] auto array_object() const -> nanobind::object {
    return std::visit([](const auto& g) { return g.array_object(); }, value_);
  }

  /// @brief Get the state for pickling.
  /// @return Tuple representing the state (axes..., array).
  [[nodiscard]] auto getstate() const -> nanobind::tuple {
    const auto n = ndim();
    nanobind::list items;
    for (size_t i = 0; i < n; ++i) {
      items.append(pybind_axis_object(i));
    }
    items.append(array_object());
    return nanobind::tuple(items);
  }

  /// @brief Get a string representation of the grid.
  /// @return String representation.
  [[nodiscard]] auto repr() const -> std::string {
    return std::visit([](const auto& g) { return g.repr(); }, value_);
  }

  /// @brief Get the underlying variant.
  /// @return Reference to the variant.
  [[nodiscard]] auto variant() const noexcept -> const GridVariant& {
    return value_;
  }

  /// @brief Get mutable access to the underlying variant.
  /// @return Mutable reference to the variant.
  [[nodiscard]] auto variant() noexcept -> GridVariant& { return value_; }

  /// @brief Visit the underlying grid with a visitor.
  /// @tparam Visitor The visitor type.
  /// @param visitor The visitor callable.
  /// @return The result of visiting.
  template <typename Visitor>
  [[nodiscard]] auto visit(Visitor&& visitor) const {
    return std::visit(std::forward<Visitor>(visitor), value_);
  }

  /// @brief Visit the underlying grid with a visitor (mutable).
  /// @tparam Visitor The visitor type.
  /// @param visitor The visitor callable.
  /// @return The result of visiting.
  template <typename Visitor>
  [[nodiscard]] auto visit(Visitor&& visitor) {
    return std::visit(std::forward<Visitor>(visitor), value_);
  }

  /// @brief Check if the holder contains a specific grid type.
  /// @tparam GridType The grid type to check for.
  /// @return True if the holder contains the specified type.
  template <typename GridType>
  [[nodiscard]] auto holds() const noexcept -> bool {
    return std::holds_alternative<GridType>(value_);
  }

  /// @brief Get the grid as a specific type.
  /// @tparam GridType The grid type to get.
  /// @return Reference to the grid.
  /// @throws std::bad_variant_access if the type doesn't match.
  template <typename GridType>
  [[nodiscard]] auto as() const -> const GridType& {
    return std::get<GridType>(value_);
  }

  /// @brief Get the grid as a specific type (mutable).
  /// @tparam GridType The grid type to get.
  /// @return Mutable reference to the grid.
  /// @throws std::bad_variant_access if the type doesn't match.
  template <typename GridType>
  [[nodiscard]] auto as() -> GridType& {
    return std::get<GridType>(value_);
  }

  /// @brief Try to get the grid as a specific type.
  /// @tparam GridType The grid type to get.
  /// @return Pointer to the grid, or nullptr if type doesn't match.
  template <typename GridType>
  [[nodiscard]] auto try_as() const noexcept -> const GridType* {
    return std::get_if<GridType>(&value_);
  }

  /// @brief Try to get the grid as a specific type (mutable).
  /// @tparam GridType The grid type to get.
  /// @return Pointer to the grid, or nullptr if type doesn't match.
  template <typename GridType>
  [[nodiscard]] auto try_as() noexcept -> GridType* {
    return std::get_if<GridType>(&value_);
  }

 private:
  GridVariant value_;
};

/// @brief Factory to create grids from Python with runtime dtype detection.
/// @tparam MathAxes Axis types (math::Axis<double>, math::TemporalAxis, etc.).
/// @param axes Tuple of axes.
/// @param array_obj Python array object.
/// @return GridHolder containing the created grid.
template <AxisTypeConcept... MathAxes>
  requires(sizeof...(MathAxes) >= 1 && sizeof...(MathAxes) <= 4)
auto grid_factory(detail::pybind_axes_tuple_t<MathAxes...>&& axes,
                  const nanobind::object& array_obj) -> GridHolder {
  // Helper to create the grid with the correct DataType
  auto create_grid = [&]<typename DataType>() -> GridHolder {
    // Cast the Python object to the specific DataType array
    using array_t = nanobind::ndarray<nanobind::numpy, DataType,
                                      nanobind::ndim<sizeof...(MathAxes)>,
                                      nanobind::c_contig>;

    // Extract axes from tuple and create grid
    return std::apply(
        [&](auto&&... ax) -> GridHolder {
          return GridHolder(
              Grid<DataType, MathAxes...>(std::forward<decltype(ax)>(ax)...,
                                          nanobind::cast<array_t>(array_obj)));
        },
        std::move(axes));
  };

  // Get a generic ndarray view to check dtype
  using generic_array_t =
      nanobind::ndarray<nanobind::numpy, nanobind::c_contig,
                        nanobind::ndim<sizeof...(MathAxes)>>;
  auto array = nanobind::cast<generic_array_t>(array_obj);
  const auto dtype = array.dtype();

  if (dtype == nanobind::dtype<int8_t>()) {
    return create_grid.template operator()<int8_t>();
  }
  if (dtype == nanobind::dtype<uint8_t>()) {
    return create_grid.template operator()<uint8_t>();
  }
  if (dtype == nanobind::dtype<int16_t>()) {
    return create_grid.template operator()<int16_t>();
  }
  if (dtype == nanobind::dtype<uint16_t>()) {
    return create_grid.template operator()<uint16_t>();
  }
  if (dtype == nanobind::dtype<int32_t>()) {
    return create_grid.template operator()<int32_t>();
  }
  if (dtype == nanobind::dtype<uint32_t>()) {
    return create_grid.template operator()<uint32_t>();
  }
  if (dtype == nanobind::dtype<int64_t>()) {
    return create_grid.template operator()<int64_t>();
  }
  if (dtype == nanobind::dtype<uint64_t>()) {
    return create_grid.template operator()<uint64_t>();
  }
  if (dtype == nanobind::dtype<float>()) {
    return create_grid.template operator()<float>();
  }
  if (dtype == nanobind::dtype<double>()) {
    return create_grid.template operator()<double>();
  }

  throw std::invalid_argument("Unsupported array dtype");
}

/// Bind Grid classes to Python.
auto init_grids(nanobind::module_& m) -> void;

}  // namespace pyinterp::pybind
