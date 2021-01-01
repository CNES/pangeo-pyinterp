// Copyright (c) 2020 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/storage/unqlite.hpp"

#include <snappy.h>

#include <algorithm>

namespace pyinterp::storage::unqlite {

struct Slice {
  char* ptr;
  Py_ssize_t len;

  explicit Slice(const pybind11::object& value)
      : ptr(PyBytes_AS_STRING(value.ptr())),
        len(PyBytes_GET_SIZE(value.ptr())) {}
};

// ---------------------------------------------------------------------------
auto Database::handle_rc(const int rc) -> void {
  // No errors detected, we have nothing more to do.
  if (rc == UNQLITE_OK) {
    return;
  }

  // If the log reading has failed, a static error message is returned
  switch (rc) {
    case UNQLITE_NOMEM:
      throw OperationalError("Out of memory");
    case UNQLITE_ABORT:
      throw OperationalError("Another thread have released this instance");
    case UNQLITE_IOERR:
      throw OperationalError("IO error");
    case UNQLITE_CORRUPT:
      throw OperationalError("Corrupt pointer");
    case UNQLITE_LOCKED:
      throw LockError("Forbidden operation");
    case UNQLITE_BUSY:
      throw LockError("The database file is locked");
    case UNQLITE_DONE:
      throw OperationalError("Operation done");
    case UNQLITE_PERM:
      throw OperationalError("Permission error");
    case UNQLITE_NOTIMPLEMENTED:
      throw ProgrammingError(
          "Method not implemented by the underlying Key/Value storage engine");
    case UNQLITE_NOTFOUND:
      throw ProgrammingError("No such record");
    case UNQLITE_NOOP:
      throw ProgrammingError("No such method");
    case UNQLITE_INVALID:
      throw ProgrammingError("Invalid parameter");
    case UNQLITE_EOF:
      throw OperationalError("End Of Input");
    case UNQLITE_UNKNOWN:
      throw ProgrammingError("Unknown configuration option");
    case UNQLITE_LIMIT:
      throw OperationalError("Database limit reached");
    case UNQLITE_EXISTS:
      throw OperationalError("Record exists");
    case UNQLITE_EMPTY:
      throw ProgrammingError("Empty record");
    case UNQLITE_COMPILE_ERR:
      throw ProgrammingError("Compilation error");
    case UNQLITE_VM_ERR:
      throw OperationalError("Virtual machine error");
    case UNQLITE_FULL:
      throw OperationalError("Full database");
    case UNQLITE_CANTOPEN:
      throw ProgrammingError("Unable to open the database file");
    case UNQLITE_READ_ONLY:
      throw ProgrammingError("Read only Key/Value storage engine");
    case UNQLITE_LOCKERR:
      throw OperationalError("Locking protocol error");
    default:
      break;
  }
  throw DatabaseError("Unknown error code. (" + std::to_string(rc) + ")");
}

// ---------------------------------------------------------------------------
auto Database::error_log() const -> std::string {
  const char* buffer;
  int length;

  handle_rc(unqlite_config(handle_, UNQLITE_CONFIG_ERR_LOG, &buffer, &length));
  if (length > 0) {
    return buffer;
  }
  return {};
}

// ---------------------------------------------------------------------------
// Parsing the string representing the reading mode of the database.
static auto decode_mode(const std::string& mode) -> unsigned int {
  auto appending = int(0);
  auto reading = int(0);
  auto memory = int(0);
  auto writing = int(0);

  for (const auto& item : mode) {
    switch (item) {
      case 'a':
        appending = 1;
        break;
      case 'r':
        reading = 1;
        break;
      case 'w':
        writing = 1;
        break;
      case 'm':
        memory = 1;
        break;
      default:
        throw std::invalid_argument("invalid mode: " + mode);
    }
    if (std::count(mode.begin(), mode.end(), item) != 1) {
      throw std::invalid_argument("invalid mode: " + mode);
    }
  }

  if (appending + reading + writing > 1) {
    throw std::invalid_argument(
        "must have exactly one of append/read/write mode");
  }

  if (memory + appending + writing > 1) {
    throw std::invalid_argument("mode 'm' can be combined anly with 'r'");
  }

  auto result = static_cast<unsigned int>(0);
  if (writing != 0) {
    result |= UNQLITE_OPEN_CREATE;
  }
  if (appending != 0) {
    result |= UNQLITE_OPEN_READWRITE;
  }
  if (reading != 0) {
    result |= UNQLITE_OPEN_READONLY;
  }
  if (memory != 0) {
    result |= UNQLITE_OPEN_MMAP;
  }
  return result;
}

// ---------------------------------------------------------------------------
Database::Database(std::string name,
                   const std::optional<std::string>& open_mode,
                   const CompressionType compression_type)
    : name_(std::move(name)),
      open_mode_(open_mode.value_or("rm")),
      compression_type_(compression_type) {
  auto mode = decode_mode(open_mode_);
  handle_rc(unqlite_open(&handle_, name_.c_str(), mode));
}

// ---------------------------------------------------------------------------
Database::~Database() {
  try {
    handle_rc(unqlite_close(handle_));
  } catch (std::runtime_error& ex) {
    PyErr_WarnEx(PyExc_RuntimeWarning, ex.what(), 1);
  }
}

// ---------------------------------------------------------------------------
static auto no_compress(const Slice& slice) -> pybind11::bytes {
  auto result = pybind11::reinterpret_steal<pybind11::bytes>(
      PyBytes_FromStringAndSize(nullptr, slice.len + 1));
  auto* buffer = PyBytes_AS_STRING(result.ptr());

  // We store the type of compression used
  buffer[0] = kNoCompression;
  ++buffer;

  memcpy(buffer, slice.ptr, sizeof(*buffer) * slice.len);
  return result;
}

// ---------------------------------------------------------------------------
auto Database::getstate() const -> pybind11::tuple {
  if (name_ == ":mem:") {
    throw std::runtime_error("Cannot pickle in-memory databases");
  }
  return pybind11::make_tuple(name_, open_mode_, compression_type_);
}

// ---------------------------------------------------------------------------
auto Database::setstate(const pybind11::tuple& state)
    -> std::shared_ptr<Database> {
  if (pybind11::len(state) != 3) {
    throw std::invalid_argument("invalid state");
  }
  return std::make_shared<Database>(state[0].cast<std::string>(),
                                    state[1].cast<std::string>(),
                                    state[2].cast<CompressionType>());
}

// ---------------------------------------------------------------------------
static auto snappy_compress(const Slice& slice) -> pybind11::bytes {
  auto compressed_len = snappy::MaxCompressedLength(slice.len);
  auto result = pybind11::reinterpret_steal<pybind11::bytes>(
      PyBytes_FromStringAndSize(nullptr, compressed_len + 1));
  auto* buffer = PyBytes_AS_STRING(result.ptr());

  // We store the type of compression used
  buffer[0] = kSnappyCompression;
  ++buffer;

  {
    auto gil = pybind11::gil_scoped_release();
    snappy::RawCompress(slice.ptr, slice.len, buffer, &compressed_len);
  }
  if (_PyBytes_Resize(&result.ptr(), compressed_len + 1) < 0) {
    throw pybind11::error_already_set();
  }
  return result;
}

// ---------------------------------------------------------------------------
static auto snappy_uncompress(const Slice& slice) -> pybind11::bytes {
  auto uncompressed_len = size_t(0);
  if (!snappy::GetUncompressedLength(slice.ptr + 1, slice.len - 1,
                                     &uncompressed_len)) {
    throw OperationalError("unable to uncompress data");
  }
  auto result = pybind11::reinterpret_steal<pybind11::bytes>(
      PyBytes_FromStringAndSize(nullptr, uncompressed_len + 1));
  {
    auto gil = pybind11::gil_scoped_release();
    snappy::RawUncompress(slice.ptr + 1, static_cast<size_t>(slice.len - 1),
                          PyBytes_AS_STRING(result.ptr()));
  }
  return result;
}

// ---------------------------------------------------------------------------
static auto no_uncompress(const Slice& slice) -> pybind11::bytes {
  auto result = pybind11::reinterpret_steal<pybind11::bytes>(
      PyBytes_FromStringAndSize(nullptr, slice.len - 1));
  auto* buffer = PyBytes_AS_STRING(result.ptr());
  memcpy(buffer, slice.ptr + 1, slice.len - 1);
  return result;
}

// ---------------------------------------------------------------------------
auto Database::compress(const pybind11::bytes& bytes) const -> pybind11::bytes {
  auto slice = Slice(bytes);
  switch (compression_type_) {
    case kNoCompression:
      return no_compress(slice);
      break;
    case kSnappyCompression:
      return snappy_compress(slice);
      break;
  }
  throw OperationalError("unknown compression type " +
                         std::to_string(compression_type_));
}

// ---------------------------------------------------------------------------
auto Database::uncompress(const pybind11::bytes& bytes) -> pybind11::bytes {
  auto slice = Slice(bytes);
  if (slice.len < 2) {
    throw OperationalError("unable to uncompress value");
  }
  switch (slice.ptr[0]) {
    case kNoCompression:
      return no_uncompress(slice);
      break;
    case kSnappyCompression:
      return snappy_uncompress(slice);
      break;
  }
  throw OperationalError("unknown compression type " +
                         std::to_string(static_cast<int>(slice.ptr[0])));
}

// ---------------------------------------------------------------------------
auto Database::setitem(const pybind11::bytes& key,
                       const pybind11::object& obj) const -> void {
  pybind11::list value;
  if (PyList_Check(obj.ptr())) {
    value = obj;
  } else {
    value.append(obj);
  }
  auto data = compress(marshaller_.dumps(value));
  auto slice_key = Slice(key);
  auto slice_data = Slice(data);
  {
    auto gil = pybind11::gil_scoped_release();
    handle_rc(unqlite_kv_store(handle_, slice_key.ptr,
                               static_cast<int>(slice_key.len), slice_data.ptr,
                               static_cast<unqlite_int64>(slice_data.len)));
  }
}

// ---------------------------------------------------------------------------
auto Database::update(const pybind11::iterable& other) const -> void {
  try {
    for (const auto item : other) {
      auto pair = item.cast<std::pair<pybind11::bytes, pybind11::object>>();
      setitem(pybind11::reinterpret_borrow<pybind11::object>(pair.first),
              pybind11::reinterpret_borrow<pybind11::object>(pair.second));
    }
  } catch (pybind11::cast_error&) {
    throw std::invalid_argument(
        "other must by an iterable of Tuple[bytes, Any]");
  }
}

// ---------------------------------------------------------------------------
auto Database::getitem(const pybind11::bytes& key) const -> pybind11::list {
  auto* const ptr_key = PyBytes_AS_STRING(key.ptr());
  unqlite_int64 size;

  auto rc = unqlite_kv_fetch(handle_, ptr_key, -1, nullptr, &size);
  if (rc == UNQLITE_NOTFOUND) {
    return pybind11::list();
  }
  if (rc != UNQLITE_OK) {
    handle_rc(rc);
  }
  auto data = pybind11::reinterpret_steal<pybind11::bytes>(
      PyBytes_FromStringAndSize(nullptr, size));
  if (data.ptr() == nullptr) {
    throw std::runtime_error("out of memory");
  }
  auto* buffer = PyBytes_AS_STRING(data.ptr());
  {
    auto gil = pybind11::gil_scoped_release();
    handle_rc(unqlite_kv_fetch(handle_, ptr_key, -1, buffer, &size));
  }
  return marshaller_.loads(uncompress(data));
}

// ---------------------------------------------------------------------------
auto Database::extend(const pybind11::iterable& other) const -> void {
  try {
    for (const auto item : other) {
      auto pair = item.cast<std::pair<pybind11::bytes, pybind11::object>>();
      auto existing_value =
          getitem(pybind11::reinterpret_borrow<pybind11::object>(pair.first));
      // The key does not exist, so an insertion is made.
      if (pybind11::len(existing_value) == 0) {
        setitem(pybind11::reinterpret_borrow<pybind11::object>(pair.first),
                pybind11::reinterpret_borrow<pybind11::object>(pair.second));
      } else {
        if (PyList_Check(pair.second.ptr())) {
          // If the value to be inserted is a list, it's concatenated with the
          // existing data.
          auto cat = pybind11::reinterpret_steal<pybind11::object>(
              PySequence_InPlaceConcat(existing_value.ptr(),
                                       pair.second.ptr()));
          setitem(pybind11::reinterpret_borrow<pybind11::object>(pair.first),
                  pybind11::reinterpret_borrow<pybind11::object>(cat));
        } else {
          // Otherwise the new data is inserted in the existing list.
          existing_value.append(pair.second);
          setitem(
              pybind11::reinterpret_borrow<pybind11::object>(pair.first),
              pybind11::reinterpret_borrow<pybind11::object>(existing_value));
        }
      }
    }
  } catch (pybind11::cast_error&) {
    throw std::invalid_argument(
        "other must by an iterable of Tuple[bytes, Any]");
  }
}

// ---------------------------------------------------------------------------
auto Database::values(const std::optional<pybind11::list>& keys) const
    -> pybind11::list {
  auto result = pybind11::list();
  for (const auto key : keys.has_value() ? keys.value() : this->keys()) {
    if (!PyBytes_Check(key.ptr())) {
      throw std::invalid_argument("key must be bytes: " +
                                  std::string(pybind11::repr(key)));
    }
    result.append(getitem(pybind11::reinterpret_borrow<pybind11::bytes>(key)));
  }
  return result;
}

// ---------------------------------------------------------------------------
auto Database::items(const std::optional<pybind11::list>& keys) const
    -> pybind11::list {
  auto result = pybind11::list();
  for (const auto key : keys.has_value() ? keys.value() : this->keys()) {
    if (!PyBytes_Check(key.ptr())) {
      throw std::invalid_argument("key must be bytes: " +
                                  std::string(pybind11::repr(key)));
    }
    result.append(pybind11::make_tuple(
        key, getitem(pybind11::reinterpret_borrow<pybind11::bytes>(key))));
  }
  return result;
}

// ---------------------------------------------------------------------------
auto Database::keys() const -> pybind11::list {
  int key_len;
  auto result = pybind11::list();

  unqlite_kv_cursor* cursor = nullptr;
  handle_rc(unqlite_kv_cursor_init(handle_, &cursor));

  try {
    for (unqlite_kv_cursor_first_entry(cursor);
         unqlite_kv_cursor_valid_entry(cursor) != 0;
         unqlite_kv_cursor_next_entry(cursor)) {
      handle_rc(unqlite_kv_cursor_key(cursor, nullptr, &key_len));

      auto item = pybind11::reinterpret_steal<pybind11::bytes>(
          PyBytes_FromStringAndSize(nullptr, key_len));
      handle_rc(unqlite_kv_cursor_key(cursor, PyBytes_AS_STRING(item.ptr()),
                                      &key_len));
      result.append(item);
    }
    unqlite_kv_cursor_release(handle_, cursor);
  } catch (...) {
    unqlite_kv_cursor_release(handle_, cursor);
    throw;
  }
  return result;
}

// ---------------------------------------------------------------------------
auto Database::len() const -> size_t {
  auto result = size_t(0);

  unqlite_kv_cursor* cursor = nullptr;
  handle_rc(unqlite_kv_cursor_init(handle_, &cursor));

  try {
    for (unqlite_kv_cursor_first_entry(cursor);
         unqlite_kv_cursor_valid_entry(cursor) != 0;
         unqlite_kv_cursor_next_entry(cursor)) {
      ++result;
    }
    unqlite_kv_cursor_release(handle_, cursor);
  } catch (...) {
    unqlite_kv_cursor_release(handle_, cursor);
    throw;
  }
  return result;
}

// ---------------------------------------------------------------------------
auto Database::clear() const -> void {
  unqlite_kv_cursor* cursor = nullptr;
  auto key_len = int(0);
  auto keys = std::vector<std::string>();

  handle_rc(unqlite_kv_cursor_init(handle_, &cursor));
  try {
    for (unqlite_kv_cursor_first_entry(cursor);
         unqlite_kv_cursor_valid_entry(cursor) != 0;
         unqlite_kv_cursor_next_entry(cursor)) {
      handle_rc(unqlite_kv_cursor_key(cursor, nullptr, &key_len));
      auto key = std::string(key_len + 1, '\0');
      handle_rc(unqlite_kv_cursor_key(cursor, key.data(), &key_len));
      keys.emplace_back(key);
    }
    unqlite_kv_cursor_release(handle_, cursor);
  } catch (...) {
    unqlite_kv_cursor_release(handle_, cursor);
    throw;
  }

  handle_rc(unqlite_begin(handle_));
  for (const auto& key : keys) {
    handle_rc(unqlite_kv_delete(handle_, key.data(), -1));
  }
  handle_rc(unqlite_commit(handle_));
}

// ---------------------------------------------------------------------------
auto Database::delitem(const pybind11::bytes& key) const -> void {
  int rc = unqlite_kv_delete(handle_, PyBytes_AS_STRING(key.ptr()), -1);
  if (rc == UNQLITE_NOTFOUND) {
    throw std::out_of_range(PyBytes_AS_STRING(key.ptr()));
  }
  if (rc != UNQLITE_OK) {
    handle_rc(rc);
  }
}

// ---------------------------------------------------------------------------
auto Database::commit() const -> void { handle_rc(unqlite_commit(handle_)); }

// ---------------------------------------------------------------------------
auto Database::rollback() const -> void {
  handle_rc(unqlite_rollback(handle_));
}

// ---------------------------------------------------------------------------
auto Database::contains(const pybind11::bytes& key) const -> bool {
  auto size = unqlite_int64(0);
  auto rc = unqlite_kv_fetch(handle_, PyBytes_AS_STRING(key.ptr()), -1, nullptr,
                             &size);
  if (rc == UNQLITE_NOTFOUND) {
    return false;
  }
  if (rc == UNQLITE_OK) {
    return true;
  }
  handle_rc(rc);
  return false;  // suppress warn from the compiler
}

}  // namespace pyinterp::storage::unqlite