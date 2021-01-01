// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geohash/utility.hpp"

namespace py = pybind11;

void init_geohash_utility(py::module& m) {
  m.def("update_dict", &pyinterp::update_dict, py::arg("dictionary"),
        py::arg("others"),
        R"__doc__(
Updates the dictionary from the "others" tuple containing the keys/values to
be taken into account by either adding or concatenating existing dictionary
values.

Args:
    dictionary (dict): Dictionnary to update.
    others (iterable): Iterable of tuples that contains Key/value to process.
)__doc__");
}
