// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <format>
#include <stdexcept>
#include <string_view>

namespace pyinterp::geometry::pybind {

/// @brief Template proxy view for container access with Python semantics.
///
/// This template class provides a reusable proxy for exposing container-like
/// access patterns through nanobind. It handles bounds checking and converts
/// negative indices appropriately. Supports two access patterns:
/// 1. Container-based: where a getter returns the actual container reference
/// 2. Index-based: where size, get, set, append, and clear are provided as
///    callable traits
///
/// @tparam ContainerOwner The type that owns the container
/// @tparam ElementType The type of elements in the container
/// @tparam Traits A struct/class providing the access interface with methods:
/// - size_getter: returns size_t given ContainerOwner*
/// - item_getter: returns ElementType& given ContainerOwner* and size_t index
/// - item_setter: sets element at given ContainerOwner* and size_t index
/// - appender: appends element to ContainerOwner*
/// - clearer: clears container in ContainerOwner*
template <typename ContainerOwner, typename ElementType, typename Traits>
class ContainerView {
 public:
  /// @brief Construct a view from an owner.
  /// @param[in] owner Pointer to the container owner.
  /// @param[in] error_msg Error message for out-of-range access.
  explicit ContainerView(ContainerOwner* owner, std::string_view error_msg)
      : owner_(owner), error_msg_(error_msg) {}

  /// @brief Number of elements in the container.
  [[nodiscard]] auto size() const -> size_t {
    return Traits::size_getter(owner_);
  }

  /// @brief Get element at index (mutable).
  /// @param[in] idx Zero-based index or negative for reverse indexing.
  /// @return Reference to element at index.
  auto get(Eigen::Index idx) -> ElementType& {
    const auto sz = static_cast<Eigen::Index>(size());

    // Handle negative indices
    if (idx < 0) {
      idx += sz;
    }

    if (idx < 0 || idx >= sz) {
      throw std::out_of_range(std::string(error_msg_));
    }

    return Traits::item_getter(owner_, static_cast<size_t>(idx));
  }

  /// @brief Set element at index.
  /// @param[in] idx Zero-based index or negative for reverse indexing.
  /// @param[in] element New element value.
  void set(Eigen::Index idx, const ElementType& element) {
    const auto sz = static_cast<Eigen::Index>(size());

    // Handle negative indices
    if (idx < 0) {
      idx += sz;
    }

    if (idx < 0 || idx >= sz) {
      throw std::out_of_range(std::string(error_msg_));
    }

    Traits::item_setter(owner_, static_cast<size_t>(idx), element);
  }

  /// @brief Append an element to the container.
  /// @param[in] element Element to append.
  void append(const ElementType& element) { Traits::appender(owner_, element); }

  /// @brief Remove all elements from the container.
  void clear() { Traits::clearer(owner_); }

 private:
  ContainerOwner* owner_;
  std::string_view error_msg_;
};

/// @brief Template function to bind a ContainerView type to nanobind.
///
/// This function simplifies binding proxy views to Python by automatically
/// generating all the necessary method bindings (__len__, __getitem__, etc.)
///
/// @tparam ViewType The ContainerView type to bind
/// @tparam ElementType The element type in the container
/// @param[in] m The nanobind module to bind to
/// @param[in] class_name Python class name for the view
/// @param[in] element_name Human-readable name for elements
template <typename ViewType, typename ElementType>
void bind_container_view(nanobind::module_& m, std::string_view class_name,
                         std::string_view element_name) {
  using nanobind::literals::operator""_a;

  nanobind::class_<ViewType>(m, class_name.data())
      .def("__len__", &ViewType::size,
           std::format("Number of {}s.", element_name).c_str())
      .def(
          "__getitem__",
          [](ViewType& view, Eigen::Index idx) -> ElementType& {
            return view.get(idx);
          },
          nanobind::arg("index"), nanobind::rv_policy::reference_internal,
          "Get element at index.")
      .def(
          "__setitem__",
          [](ViewType& view, Eigen::Index idx, const ElementType& elem) {
            view.set(idx, elem);
          },
          nanobind::arg("index"), nanobind::arg("item"),
          "Set element at index.")
      .def("append", &ViewType::append, "Add an element.",
           nanobind::arg("item"))
      .def("clear", &ViewType::clear, "Remove all elements.")
      .def("__iter__", [](ViewType& view) {
        nanobind::list items;
        for (size_t i = 0; i < view.size(); ++i) {
          items.append(view.get(static_cast<Eigen::Index>(i)));
        }
        return items.attr("__iter__")();
      });
}

}  // namespace pyinterp::geometry::pybind
