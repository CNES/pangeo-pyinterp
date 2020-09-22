// Copyright (c) 2020 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/storage/unqlite.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace storage = pyinterp::storage::unqlite;
namespace py = pybind11;

void init_storage_unqlite(py::module& m) {
  auto submodule = m.def_submodule("unqlite", R"__doc__(

UnQLite Key/Value Storage
-------------------------
)__doc__");

  py::register_exception<storage::DatabaseError>(submodule, "DatabaseError",
                                                 PyExc_RuntimeError);
  py::register_exception<storage::ProgrammingError>(
      submodule, "ProgrammingError", PyExc_RuntimeError);
  py::register_exception<storage::OperationalError>(
      submodule, "OperationalError", PyExc_RuntimeError);
  py::register_exception<storage::LockError>(submodule, "LockError",
                                             PyExc_IOError);

  py::enum_<storage::CompressionType>(submodule, "CompressionType", R"__doc__(
Known compression algorithms used to compress the values stored in the
database.
)__doc__")
      .value("none", storage::kNoCompression, "No commpression")
      .value("snappy", storage::kSnappyCompression,
             "Compress values with Snappy");

  py::class_<storage::Database, std::shared_ptr<storage::Database>>(
      submodule, "Database", "Key/Value store")
      .def(py::init<std::string, const std::optional<std::string>&,
                    storage::CompressionType>(),
           py::arg("name"), py::arg("mode") = py::none(),
           py::arg("compression_type") = storage::kSnappyCompression,
           R"(Opening a database

Args:
     name (str): path to the target database file. If name is ":mem:" then
          a private, in-memory database is created
     mode (str, optional): optional string that specifies the mode in which
          the database is opened. Default to ``r`` which means open for
          readind. The available mode are:

          ========= ========================================================
          Character Meaning
          ========= ========================================================
          ``'r'``   open for reading (default)
          ``'w'``   open for reading/writing. Create the database if needed
          ``'a'``   open for writing. Database file must be existing.
          ``'m'``   open in read-only memory view of the whole database
          ========= ========================================================

          Mode ``m`` works only in conjunction with ``r`` mode.
     compression_mode (CompressionMode, optional): Type of compression used
          to compress values stored into the database. Only has an effect for
          new data written in the database.
)")
      .def(py::pickle(
          [](const storage::Database& self) -> py::tuple {
            return self.getstate();
          },
          [](const py::tuple& state) -> std::shared_ptr<storage::Database> {
            return storage::Database::setstate(state);
          }))
      .def("__setitem__", &storage::Database::setitem, py::arg("key"),
           py::arg("value"))
      .def("__getitem__", &storage::Database::getitem, py::arg("key"))
      .def("__delitem__", &storage::Database::delitem, py::arg("key"))
      .def("__len__", &storage::Database::len,
           py::call_guard<py::gil_scoped_release>())
      .def("__contains__", &storage::Database::contains, py::arg("key"))
      .def("error_log", &storage::Database::error_log,
           R"__doc__(
Reads the contents of the database error log

Return:
  str: The contents of the error log
)__doc__",
           py::call_guard<py::gil_scoped_release>())
      .def("commit", &storage::Database::commit,
           "Commit all changes to the database.",
           py::call_guard<py::gil_scoped_release>())
      .def("rollback", &storage::Database::rollback,
           "Rollback a write-transaction.",
           py::call_guard<py::gil_scoped_release>())
      .def("clear", &storage::Database::clear,
           "Remove all entries from the database.",
           py::call_guard<py::gil_scoped_release>())
      .def("keys", &storage::Database::keys,
           R"__doc__(
Return a list containing all the keys from the database.

Return:
    list: Keys registered in the database.
)__doc__")
      .def("update", &storage::Database::update, py::arg("map"),
           R"__doc__(
Update the database with the key/value pairs from map, overwriting"
existing keys.

Args:
    map (dict): The keys associated with the values to be stored.
Raises:
    ValueError: if the keys are not bytes
)__doc__")

      .def("extend", &storage::Database::extend, py::arg("map"),
           R"__doc__(
Extend or create the database with the key/value pairs from map

Args:
    map (dict): The keys associated with the values to be stored.
Raises:
    ValueError: if the keys are not bytes
)__doc__")
      .def("values", &storage::Database::values, py::arg("keys") = py::none(),
           R"__doc__(
Read all values from the database for the keys provided

Args:
    keys (list, optional): The keys to be read. If None, all keys stored in
        the database are read.
Return:
    list: The values associated with the keys read.
Raises:
    ValueError: if the keys are not bytes
)__doc__");
}