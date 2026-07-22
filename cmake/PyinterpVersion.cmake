# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
# Resolve the project version from the most reliable source available.
#
# The sources are tried in the order below; the first one that yields a value
# wins. Every acquisition mode of the project is covered by at least one of
# them, so configuration never depends on a single mechanism being present:
#
# 1. ``-DPYINTERP_VERSION=...`` on the command line. ``setup.py`` passes the
#    version it resolved itself, so CMake and setuptools_scm can never disagree.
# 2. ``git describe``. Authoritative in a developer checkout and in a
#    FetchContent ``GIT_REPOSITORY`` download.
# 3. ``VERSION.txt``. Contains a ``$Format:...$`` placeholder expanded by ``git
#    archive``, which covers GitHub source tarballs and FetchContent ``URL``
#    downloads -- neither has a ``.git`` directory.
# 4. ``pyinterp/_version.py``. Written by setuptools_scm and shipped in the
#    sdist.
# 5. ``PKG-INFO``. The sdist metadata, as a last resort.
#
# ``PYINTERP_VERSION`` is the input of the resolution; the results are written
# to distinct names so that a second call cannot mistake the output of the first
# one for a user-supplied override. ``pyinterp_resolve_version()`` sets in its
# caller's scope:
#
# * ``PYINTERP_VERSION_STRING`` -- normalized ``major.minor.patch``, suitable
#   for ``project(VERSION ...)``, which only accepts numeric components.
# * ``PYINTERP_VERSION_FULL`` -- the descriptive string as found, which may
#   carry a pre-release or commit suffix (``2026.6.0-1-gdeadbee``).
# * ``PYINTERP_VERSION_SOURCE`` -- which of the sources above was used, so the
#   configuration summary can show it.

include_guard(GLOBAL)

# Version used when every source failed. Deliberately invalid-looking so a
# broken build is obvious rather than silently mislabelled.
set(PYINTERP_VERSION_FALLBACK "0.0.0")

# Extract a numeric "major.minor.patch" triplet from an arbitrary version
# string. Sets ``out_var`` to the empty string when nothing can be parsed.
function(_pyinterp_normalize_version raw out_var)
  if("${raw}" MATCHES "^[vV]?([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(${out_var}
        "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}"
        PARENT_SCOPE)
  elseif("${raw}" MATCHES "^[vV]?([0-9]+)\\.([0-9]+)")
    set(${out_var}
        "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.0"
        PARENT_SCOPE)
  elseif("${raw}" MATCHES "^[vV]?([0-9]+)")
    set(${out_var}
        "${CMAKE_MATCH_1}.0.0"
        PARENT_SCOPE)
  else()
    set(${out_var}
        ""
        PARENT_SCOPE)
  endif()
endfunction()

# Source 2: ``git describe`` against the annotated release tags.
function(_pyinterp_version_from_git source_dir out_var)
  set(${out_var}
      ""
      PARENT_SCOPE)

  # ``.git`` is a directory in a normal clone and a file in a worktree or a
  # submodule, so test for mere existence.
  if(NOT EXISTS "${source_dir}/.git")
    return()
  endif()

  find_package(Git QUIET)
  if(NOT GIT_FOUND)
    return()
  endif()

  # Release tags are plain versions ("2026.6.0"); the glob keeps unrelated tags
  # from being picked up.
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" describe --tags --match "[0-9]*"
    WORKING_DIRECTORY "${source_dir}"
    OUTPUT_VARIABLE describe
    ERROR_VARIABLE error
    RESULT_VARIABLE status
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # A shallow clone or a repository without tags fails here; that is not an
  # error, the next source takes over.
  if(NOT status EQUAL 0)
    return()
  endif()

  set(${out_var}
      "${describe}"
      PARENT_SCOPE)
endfunction()

# Source 3: VERSION.txt, once expanded by ``git archive``.
function(_pyinterp_version_from_version_file source_dir out_var)
  set(${out_var}
      ""
      PARENT_SCOPE)

  set(path "${source_dir}/VERSION.txt")
  if(NOT EXISTS "${path}")
    return()
  endif()

  file(READ "${path}" content)
  string(STRIP "${content}" content)

  # Still holding the unexpanded placeholder: this is a plain checkout, not an
  # exported tarball, so the file carries no information.
  if(content STREQUAL "" OR content MATCHES "^\\$Format:")
    return()
  endif()

  set(${out_var}
      "${content}"
      PARENT_SCOPE)
endfunction()

# Source 4: pyinterp/_version.py, as written by setuptools_scm.
function(_pyinterp_version_from_python_module source_dir out_var)
  set(${out_var}
      ""
      PARENT_SCOPE)

  set(path "${source_dir}/pyinterp/_version.py")
  if(NOT EXISTS "${path}")
    return()
  endif()

  # The generated module also declares a bare ``__version__: str`` annotation;
  # requiring the assignment sign skips it.
  file(STRINGS "${path}" lines REGEX "^__version__[ \t]*=")
  foreach(line IN LISTS lines)
    if("${line}" MATCHES "['\"]([^'\"]+)['\"]")
      set(${out_var}
          "${CMAKE_MATCH_1}"
          PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

# Source 5: PKG-INFO, present at the root of an unpacked sdist.
function(_pyinterp_version_from_pkg_info source_dir out_var)
  set(${out_var}
      ""
      PARENT_SCOPE)

  set(path "${source_dir}/PKG-INFO")
  if(NOT EXISTS "${path}")
    return()
  endif()

  file(STRINGS "${path}" lines REGEX "^Version:[ \t]*")
  foreach(line IN LISTS lines)
    string(REGEX REPLACE "^Version:[ \t]*" "" value "${line}")
    string(STRIP "${value}" value)
    if(NOT value STREQUAL "")
      set(${out_var}
          "${value}"
          PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

# Resolve the version and publish it in the caller's scope. ``source_dir`` is
# the root of the pyinterp source tree.
function(pyinterp_resolve_version source_dir)
  set(version "")
  set(origin "")

  # An explicit value always wins: it is how setup.py keeps both build systems
  # in agreement, and how a packager pins a version in an exotic environment.
  if(DEFINED PYINTERP_VERSION AND NOT "${PYINTERP_VERSION}" STREQUAL "")
    set(version "${PYINTERP_VERSION}")
    set(origin "user-provided")
  endif()

  if(version STREQUAL "")
    _pyinterp_version_from_git("${source_dir}" version)
    if(NOT version STREQUAL "")
      set(origin "git")
    endif()
  endif()

  if(version STREQUAL "")
    _pyinterp_version_from_version_file("${source_dir}" version)
    if(NOT version STREQUAL "")
      set(origin "VERSION.txt")
    endif()
  endif()

  if(version STREQUAL "")
    _pyinterp_version_from_python_module("${source_dir}" version)
    if(NOT version STREQUAL "")
      set(origin "pyinterp/_version.py")
    endif()
  endif()

  if(version STREQUAL "")
    _pyinterp_version_from_pkg_info("${source_dir}" version)
    if(NOT version STREQUAL "")
      set(origin "PKG-INFO")
    endif()
  endif()

  # Never fail the configuration over a missing version: a consumer who only
  # wants to compile the C++ library should not be blocked by it. Warn loudly
  # instead, so a mislabelled release cannot go unnoticed.
  if(version STREQUAL "")
    set(version "${PYINTERP_VERSION_FALLBACK}")
    set(origin "fallback")
    message(
      WARNING
        "Could not determine the pyinterp version: no usable git checkout, "
        "VERSION.txt, pyinterp/_version.py or PKG-INFO was found under "
        "'${source_dir}'. Falling back to ${PYINTERP_VERSION_FALLBACK}. Pass "
        "-DPYINTERP_VERSION=<version> to set it explicitly.")
  endif()

  _pyinterp_normalize_version("${version}" normalized)
  if(normalized STREQUAL "")
    message(
      WARNING "Could not parse a numeric version out of '${version}'; using "
              "${PYINTERP_VERSION_FALLBACK} for project(VERSION).")
    set(normalized "${PYINTERP_VERSION_FALLBACK}")
  endif()

  set(PYINTERP_VERSION_STRING
      "${normalized}"
      PARENT_SCOPE)
  set(PYINTERP_VERSION_FULL
      "${version}"
      PARENT_SCOPE)
  set(PYINTERP_VERSION_SOURCE
      "${origin}"
      PARENT_SCOPE)
endfunction()

# Make the resolved version reachable from a parent project.
#
# ``project(VERSION ...)`` only defines ``<name>_VERSION`` in the scope of the
# directory that calls it, so a project pulling pyinterp in via
# ``add_subdirectory()`` or ``FetchContent_MakeAvailable()`` would see nothing.
# Publishing through the cache is what actually makes the version observable
# from the outside.
function(pyinterp_publish_version)
  set(pyinterp_VERSION # cmake-lint: disable=C0103
      "${PROJECT_VERSION}"
      CACHE INTERNAL "pyinterp version (major.minor.patch)")
  set(pyinterp_VERSION_FULL # cmake-lint: disable=C0103
      "${PYINTERP_VERSION_FULL}"
      CACHE INTERNAL "pyinterp version, including any pre-release suffix")
endfunction()
