// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/storage/marshaller.hpp"

#include <pybind11/pybind11.h>

namespace storage = pyinterp::storage;
namespace py = pybind11;

void init_storage_marshaller(py::module& m) {
  py::class_<storage::Marshaller>(m, "Marshaller",
                                  "Python object serialization")
      .def(py::init(), "Default constructor")
      .def("dumps", &storage::Marshaller::dumps, py::arg("obj"),
           R"__doc__(
Return the pickled representation of an object as a bytes object.

Args:
  obj (object): Object to process
Return:
  bytes: The pickled representation of the object.
)__doc__")
      .def("loads", &storage::Marshaller::loads, py::arg("bytes_object"),
           R"__doc__(
Return the reconstituted object hierarchy of a pickled representation.

Args:
  bytes_object (bytes): Pickled representation
Return:
  object: The reconstituted object.
)__doc__");
}