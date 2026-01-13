// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <any>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace pyinterp::serialization {

/// Concept for trivially serializable types
template <typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

/// Write-only serialization buffer for creating serialized data
///
/// This class provides a straightforward interface for serializing various data
/// types (scalars, vectors, strings, and Eigen matrices) into a contiguous byte
/// buffer. The buffer is stored internally and can be extracted or accessed as
/// needed.
class Writer {
  std::vector<std::byte> buffer_;

 public:
  /// Default constructor initializes an empty buffer
  Writer() = default;

  /// Write an other Writer's contents into this buffer
  /// This writes the size followed by the data (for compatibility with vector
  /// serialization)
  /// @param[in,out] other The other Writer to append
  void write(Writer&& other) {
    auto other_buffer = std::move(other).release();
    write(other_buffer);
  }

  /// Append another Writer's contents directly without size prefix
  /// Use this for nested serialization where each component has its own format
  /// (e.g., when serializing Axis or TDigest that have their own magic numbers)
  /// @param[in,out] other The other Writer to append
  void append(Writer&& other) {
    auto other_buffer = std::move(other).release();
    buffer_.insert(buffer_.end(), other_buffer.begin(), other_buffer.end());
  }

  /// Write trivially copyable types
  /// @tparam T Type of the value (must satisfy TriviallyCopyable concept)
  /// @param[in] value The value to write to the buffer
  template <TriviallyCopyable T>
  void write(const T& value) {
    auto bytes = std::as_bytes(std::span{&value, 1});
    buffer_.insert(buffer_.end(), bytes.begin(), bytes.end());
  }

  /// Write vectors to the buffer
  /// For trivially copyable element types, this performs a single memcpy
  /// operation. For other types, elements are serialized individually.
  /// @tparam T Type of vector elements
  /// @param[in] vec The vector to write
  template <typename T>
  void write(const std::vector<T>& vec) {
    write(vec.size());
    if constexpr (TriviallyCopyable<T>) {
      auto bytes = std::as_bytes(std::span{vec});
      buffer_.insert(buffer_.end(), bytes.begin(), bytes.end());
    } else {
      for (const auto& elem : vec) {
        write(elem);
      }
    }
  }

  /// Write strings to the buffer
  /// Writes the string length followed by the string data.
  /// @param[in] str The string to write
  void write(const std::string& str) {
    write(str.size());
    auto bytes = std::as_bytes(std::span{str.data(), str.size()});
    buffer_.insert(buffer_.end(), bytes.begin(), bytes.end());
  }

  /// Write Eigen matrices or vectors to the buffer
  /// Stores the matrix dimensions (rows, cols), storage order, followed by the
  /// data. For trivially copyable scalar types with contiguous data layout,
  /// uses direct memcpy. Otherwise, elements are serialized individually.
  /// @tparam Derived Eigen type (auto-deduced from MatrixBase)
  /// @param[in] mat The matrix or vector to write
  template <typename Derived>
  void write(const Eigen::MatrixBase<Derived>& mat) {
    using Scalar = typename Derived::Scalar;

    // Write dimensions
    write(static_cast<size_t>(mat.rows()));
    write(static_cast<size_t>(mat.cols()));

    // Write storage order (0 = ColumnMajor, 1 = RowMajor)
    write(static_cast<char>(Derived::IsRowMajor ? 1 : 0));

    if constexpr (TriviallyCopyable<Scalar>) {
      // Check if the data is contiguous
      if (mat.innerStride() == 1 &&
          mat.outerStride() ==
              (Derived::IsRowMajor ? mat.cols() : mat.rows())) {
        // Contiguous data, direct copy
        auto bytes = std::as_bytes(
            std::span{mat.derived().data(), static_cast<size_t>(mat.size())});
        buffer_.insert(buffer_.end(), bytes.begin(), bytes.end());
      } else {
        // Non-contiguous, copy element-wise
        for (Eigen::Index i = 0; i < mat.rows(); ++i) {
          for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            write(mat(i, j));
          }
        }
      }
    } else {
      // Non-trivially copyable, element-wise
      for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
          write(mat(i, j));
        }
      }
    }
  }

  /// Get buffer size in bytes
  /// @return The number of bytes currently in the buffer
  [[nodiscard]] auto size() const -> size_t { return buffer_.size(); }

  /// Get raw data pointer
  /// @return Pointer to the buffer's byte data
  [[nodiscard]] auto data() const -> const std::byte* { return buffer_.data(); }

  /// Get data as a span
  /// @return A std::span view of the buffer's contents
  [[nodiscard]] auto span() const -> std::span<const std::byte> {
    return buffer_;
  }

  /// Move buffer out and clear this writer
  /// This method consumes the writer (rvalue reference) and transfers ownership
  /// of the buffer to the caller. The writer is left in a moved-from state.
  /// @return The internal buffer as a vector of bytes
  [[nodiscard]] auto release() && -> std::vector<std::byte> {
    return std::move(buffer_);
  }

  /// Clear the buffer and reset to empty state
  void clear() { buffer_.clear(); }
};

/// Read-only deserialization buffer for extracting serialized data
///
/// This class provides an interface for deserializing data from a byte buffer.
/// It manages data lifetime through type-erased ownership (for owned buffers)
/// or borrows external data (for caller-managed lifetimes). The class tracks
/// the current read position to support sequential reading operations.
class Reader {
 protected:
  /// Raw data pointer to the buffer
  const std::byte* data_ptr_ = nullptr;
  /// Total size of the buffer in bytes
  size_t size_ = 0;
  /// Current read position in the buffer
  size_t read_pos_ = 0;
  /// Type-erased owner for managing buffer lifetime in owned mode
  std::any owner_;

 public:
  /// Default constructor
  Reader() = default;

  /// Construct from a Writer, taking ownership of its buffer
  /// @param[in,out] writer A Writer instance (consumed by moving)
  explicit Reader(Writer&& writer) {
    auto vec =
        std::make_shared<std::vector<std::byte>>(std::move(writer).release());
    data_ptr_ = vec->data();
    size_ = vec->size();
    owner_ = vec;
  }

  /// Construct with owned data
  /// Takes ownership of the provided buffer. The lifetime is managed
  /// internally.
  /// @param[in] buffer Buffer to take ownership of
  explicit Reader(std::vector<std::byte> buffer) {
    auto vec = std::make_shared<std::vector<std::byte>>(std::move(buffer));
    data_ptr_ = vec->data();
    size_ = vec->size();
    owner_ = vec;
  }

  /// Construct from raw pointer with caller-managed lifetime
  /// The caller is responsible for ensuring the buffer remains valid
  /// for the lifetime of this Reader instance.
  /// @param[in] data Raw data pointer to buffer
  /// @param[in] size Size of the buffer in bytes
  Reader(const std::byte* data, size_t size) : data_ptr_(data), size_(size) {}

  /// Construct with custom owner for lifetime management
  /// The owner object is stored type-erased to manage buffer lifetime.
  /// @tparam Owner Type of the owner object (typically a smart pointer)
  /// @param[in] data Raw data pointer to buffer
  /// @param[in] size Size of the buffer in bytes
  /// @param[in,out] owner Owner object (moved) to manage buffer lifetime
  template <typename Owner>
  Reader(const std::byte* data, size_t size, Owner&& owner)
      : data_ptr_(data), size_(size), owner_(std::forward<Owner>(owner)) {}

  /// Destructor
  virtual ~Reader() = default;

  // Disable copy (to prevent accidental copies of large buffers)
  Reader(const Reader&) = delete;
  auto operator=(const Reader&) -> Reader& = delete;

  // Enable move
  Reader(Reader&&) noexcept = default;
  auto operator=(Reader&&) noexcept -> Reader& = default;

  /// Get raw data pointer
  /// @return Pointer to the buffer's byte data
  [[nodiscard]] auto data() const -> const std::byte* { return data_ptr_; }

  /// Get buffer size in bytes
  /// @return The total size of the buffer
  [[nodiscard]] auto size() const -> size_t { return size_; }

  /// Get current read position
  /// @return The current offset in the buffer
  [[nodiscard]] auto tell() const -> size_t { return read_pos_; }

  /// Seek to an absolute position in the buffer
  /// @param[in] pos The position to seek to
  /// @throws std::out_of_range if pos exceeds buffer size
  void seek(size_t pos) {
    if (pos > size_) {
      throw std::out_of_range("Seek position out of range");
    }
    read_pos_ = pos;
  }

  /// Reset read position to the beginning
  void reset() { read_pos_ = 0; }

  /// Check if this Reader owns its buffer or borrows it
  /// @return true if the Reader manages buffer lifetime, false if borrowed
  [[nodiscard]] auto has_owner() const -> bool { return owner_.has_value(); }

  /// Read trivially copyable types from buffer
  /// Advances the read position by sizeof(T).
  /// @tparam T Trivially copyable type to read
  /// @return The deserialized value
  /// @throws std::out_of_range if not enough data in buffer
  template <TriviallyCopyable T>
  [[nodiscard]] auto read() -> T {
    if (read_pos_ + sizeof(T) > size_) {
      throw std::out_of_range("Buffer overflow during read");
    }
    T value;
    std::memcpy(&value, data_ptr_ + read_pos_, sizeof(T));
    read_pos_ += sizeof(T);
    return value;
  }

  /// Read vectors from buffer
  /// Expects the buffer to contain size followed by elements.
  /// For trivially copyable types, uses direct memcpy. Otherwise, reads
  /// elements individually.
  /// @tparam T Type of vector elements
  /// @return The deserialized vector
  /// @throws std::out_of_range if not enough data in buffer
  template <typename T>
  [[nodiscard]] auto read_vector() -> std::vector<T> {
    auto size = read<size_t>();
    std::vector<T> vec(size);

    if constexpr (TriviallyCopyable<T>) {
      if (read_pos_ + size * sizeof(T) > size_) {
        throw std::out_of_range("Buffer overflow during vector read");
      }
      std::memcpy(vec.data(), data_ptr_ + read_pos_, size * sizeof(T));
      read_pos_ += size * sizeof(T);
    } else {
      for (auto& elem : vec) {
        elem = read<T>();
      }
    }
    return vec;
  }

  /// Read strings from buffer
  /// Expects the buffer to contain string length followed by character data.
  /// @return The deserialized string
  /// @throws std::out_of_range if not enough data in buffer
  [[nodiscard]] auto read_string() -> std::string {
    auto size = read<size_t>();
    if (read_pos_ + size > size_) {
      throw std::out_of_range("Buffer overflow during string read");
    }
    std::string str(reinterpret_cast<const char*>(data_ptr_ + read_pos_), size);
    read_pos_ += size;
    return str;
  }

  /// Read Eigen matrices or vectors from buffer
  /// Expects the buffer to contain dimensions (rows, cols), storage order,
  /// followed by element data. For trivially copyable scalar types, uses direct
  /// memcpy. Otherwise, reads elements individually.
  /// @tparam T Scalar type of matrix elements
  /// @tparam Rows Number of rows (Eigen::Dynamic for runtime-determined size,
  /// or fixed compile-time size)
  /// @tparam Cols Number of columns (Eigen::Dynamic for runtime-determined
  /// size, or fixed compile-time size)
  /// @return The deserialized Eigen matrix
  /// @throws std::out_of_range if not enough data in buffer
  template <typename T>
  [[nodiscard]] auto read_eigen()
      -> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {
    auto rows = static_cast<Eigen::Index>(read<size_t>());
    auto cols = static_cast<Eigen::Index>(read<size_t>());
    auto storage_order = read<char>();  // 0 = ColumnMajor, 1 = RowMajor

    using MatrixColMajor =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using MatrixRowMajor =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    if constexpr (TriviallyCopyable<T>) {
      auto total_size = static_cast<size_t>(rows * cols) * sizeof(T);
      if (read_pos_ + total_size > size_) {
        throw std::out_of_range("Buffer overflow during matrix read");
      }

      if (storage_order == 1) {
        // Read as RowMajor
        MatrixRowMajor result(rows, cols);
        std::memcpy(result.data(), data() + read_pos_, total_size);
        read_pos_ += total_size;
        return result;
      }
      // Read as ColumnMajor
      MatrixColMajor result(rows, cols);
      std::memcpy(result.data(), data() + read_pos_, total_size);
      read_pos_ += total_size;
      return result;
    }

    // Non-trivially copyable: read element-wise in the correct order
    if (storage_order == 1) {
      // RowMajor: iterate rows first
      MatrixRowMajor result(rows, cols);
      for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < cols; ++j) {
          result(i, j) = read<T>();
        }
      }
      return result;
    }
    // ColumnMajor: iterate columns first
    MatrixColMajor result(rows, cols);
    for (Eigen::Index j = 0; j < cols; ++j) {
      for (Eigen::Index i = 0; i < rows; ++i) {
        result(i, j) = read<T>();
      }
    }
    return result;
  }
};

}  // namespace pyinterp::serialization
