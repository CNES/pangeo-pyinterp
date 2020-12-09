// Copyright (c) 2020 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/pybind11.h>
#include <unqlite.h>

#include <optional>
#include <string>

#include "pyinterp/storage/marshaller.hpp"

namespace pyinterp::storage::unqlite {

/// Exception raised for errors that are related to the database
class DatabaseError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// Exception raised for programming errors
class ProgrammingError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// Exception raised for errors that are related to the database’s operation and
/// not necessarily under the control of the programmer
class OperationalError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// Exception raised for errors that are related to lock operations
class LockError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// The data stored in the database can be compressed using the following
/// algorithms.
enum CompressionType {
  kNoCompression = 0x0,
  kSnappyCompression = 0x1,
};

/// Key/Value store
class Database {
 public:
  // Default constructor
  Database(std::string name, const std::optional<std::string>& open_mode,
           CompressionType compression_type);

  /// Destructor
  virtual ~Database();

  /// Copy constructor
  Database(const Database&) = delete;

  /// Copy assignment operator
  auto operator=(const Database&) -> Database& = delete;

  /// Get state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple;

  /// Create a new instance from the information saved in the "state" variable
  static auto setstate(const pybind11::tuple& state)
      -> std::shared_ptr<Database>;

  /// Set the key/value pair, overwriting existing
  auto setitem(const pybind11::bytes& key, const pybind11::object& obj) const
      -> void;

  /// Update the database with the key/value pairs from map, overwriting
  /// existing keys
  auto update(const pybind11::iterable& other) const -> void;

  /// Extend or create the database with the key/value pairs from map
  auto extend(const pybind11::iterable& other) const -> void;

  /// Return the item of the database with key key. Return an empty list if key
  /// is not in the database.
  [[nodiscard]] auto getitem(const pybind11::bytes& key) const
      -> pybind11::list;

  /// Read all values from the database for the keys provided
  [[nodiscard]] auto values(const std::optional<pybind11::list>& keys) const
      -> pybind11::list;

  /// Return the dictionary's items ((key, value) pairs).
  [[nodiscard]] auto items(const std::optional<pybind11::list>& keys) const
      -> pybind11::list;

  /// Remove the key from the database. Raises a KeyError if key is not int the
  /// database
  auto delitem(const pybind11::bytes& key) const -> void;

  /// Return a list containing all the keys from the database
  [[nodiscard]] auto keys() const -> pybind11::list;

  /// Remove all items from the database
  auto clear() const -> void;

  /// Return the number of items in the database
  [[nodiscard]] auto len() const -> size_t;

  /// Return true if the database has a key key, else false.
  [[nodiscard]] auto contains(const pybind11::bytes& key) const -> bool;

  /// Commit all changes to the database
  auto commit() const -> void;

  /// Rollback a write-transaction
  auto rollback() const -> void;

  /// Read error log
  [[nodiscard]] auto error_log() const -> std::string;

 private:
  ::unqlite* handle_{nullptr};
  std::string name_;
  std::string open_mode_;
  Marshaller marshaller_{};
  CompressionType compression_type_;

  static auto handle_rc(int rc) -> void;

  [[nodiscard]] auto compress(const pybind11::bytes& bytes) const
      -> pybind11::bytes;
  [[nodiscard]] static auto uncompress(const pybind11::bytes& bytes)
      -> pybind11::bytes;
};

}  // namespace pyinterp::storage::unqlite