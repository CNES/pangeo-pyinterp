#pragma once
#include <string>
#include <stdexcept>

namespace pyinterp {
namespace detail {

/// Automation of vector size control to ensure that all vectors have the same
/// size.
///
/// @param name1 name of the variable containing the first vector
/// @param v1 first vector
/// @param name2 name of the variable containing the second vector
/// @param v2 second vector
/// @throw std::invalid_argument if the size of the two vectors is different
template <typename Vector1, typename Vector2>
void check_container_size(const std::string& name1, const Vector1& v1,
                          const std::string& name2, const Vector2& v2) {
  if (v1.size() != v2.size()) {
    throw std::invalid_argument(
        name1 + ", " + name2 + " could not be broadcast together with shape (" +
        std::to_string(v1.size()) + ", ) (" + std::to_string(v2.size()) +
        ", )");
  }
}

/// Vector size check function pattern.
///
/// @param name1 name of the variable containing the first vector
/// @param v1 first vector
/// @param name2 name of the variable containing the second vector
/// @param v2 second vector
/// @param args other vectors to be verified
/// @throw std::invalid_argument if the size of the two vectors is different
template <typename Vector1, typename Vector2, typename... Args>
void check_container_size(const std::string& name1, const Vector1& v1,
                          const std::string& name2, const Vector2& v2,
                          Args... args) {
  static_assert(sizeof...(Args) % 2 == 0,
                "an even number of parameters is expected");
  check_container_size(name1, v1, name2, v2);
  check_container_size(name1, v1, args...);
}

}  // namespace detail
}  // namespace pyinterp