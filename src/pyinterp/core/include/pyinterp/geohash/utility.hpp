// Copyright (c) 2020 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <pybind11/pybind11.h>

#include <utility>

namespace pyinterp {

auto update_dict(pybind11::dict& dict, const pybind11::iterable& other)
    -> void {
  try {
    pybind11::list new_value;
    for (const auto& item : other) {
      auto pair = item.cast<std::pair<pybind11::object, pybind11::object>>();
      if (PyList_Check(pair.second.ptr())) {
        new_value = pair.second;
      } else {
        new_value = pybind11::list(1);
        new_value[0] = pair.second;
      }
      if (dict.contains(pair.first)) {
        auto existing_value = dict[pair.first];
        if (PyList_Check(existing_value.ptr())) {
          new_value = pybind11::reinterpret_steal<pybind11::list>(
              PySequence_InPlaceConcat(existing_value.ptr(), new_value.ptr()));
        } else {
          new_value.insert(0, existing_value);
        }
      }
      dict[pair.first] = new_value;
    }
  } catch (pybind11::cast_error&) {
    throw std::invalid_argument("other must by an iterable of Tuple[Any, Any]");
  }
}

}  // namespace pyinterp
