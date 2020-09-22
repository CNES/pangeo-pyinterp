// Copyright (c) 2020 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/pybind11.h>

namespace pyinterp::storage {

/// Handle Python object serialization for the C++ code
class Marshaller {
 public:
  /// Default constructor
  Marshaller()
      : pickle_(pybind11::module::import("pickle")),
        dumps_(pickle_.attr("dumps")),
        loads_(pickle_.attr("loads")) {}

  /// Return the pickled representation of the object obj as a bytes object
  [[nodiscard]] inline auto dumps(const pybind11::object& obj) const
      -> pybind11::bytes {
    return dumps_(obj, -1);
  }

  /// Return the reconstituted object hierarchy of the pickled representation
  /// bytes_object of an object.
  [[nodiscard]] inline auto loads(const pybind11::bytes& bytes_object) const
      -> pybind11::object {
    return loads_(bytes_object);
  }

 private:
  pybind11::module pickle_;
  pybind11::object dumps_;
  pybind11::object loads_;
};

}  // namespace pyinterp::storage