#!/usr/bin/env python3
# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Resolve the pyinterp version from whichever source is available.

This mirrors, in Python, the resolution chain implemented in
``cmake/PyinterpVersion.cmake``. Both build systems must agree, so the sources
and their precedence are kept identical:

1. the ``PYINTERP_VERSION`` environment variable,
2. ``setuptools_scm`` against the git checkout,
3. ``VERSION.txt``, whose ``$Format:...$`` placeholder is expanded by
   ``git archive`` (GitHub source tarballs),
4. ``pyinterp/_version.py``, written by ``setuptools_scm`` into the sdist,
5. ``PKG-INFO``, the sdist metadata.

``setup.py`` passes the result down to CMake via ``-DPYINTERP_VERSION`` so the
two can never disagree.
"""

from __future__ import annotations

import os
import pathlib
import re
from collections.abc import Callable


#: Version reported when every source failed. Deliberately invalid-looking so
#: that a broken build stands out instead of shipping a plausible-but-wrong
#: number.
FALLBACK_VERSION = "0.0.0"

#: Matches the assignment written by setuptools_scm in ``_version.py``:
#: ``__version__ = version = '2026.6.0'``.
_VERSION_ASSIGNMENT = re.compile(
    r"^__version__\s*=.*?['\"]([^'\"]+)['\"]", re.MULTILINE
)

#: Matches the ``Version:`` field of an sdist ``PKG-INFO``.
_PKG_INFO_VERSION = re.compile(r"^Version:[ \t]*(.+)$", re.MULTILINE)

#: Matches the output of ``git describe``: a tag, the number of commits since
#: that tag, and the abbreviated commit, e.g. ``2026.6.0-3-gc98fef9``.
_GIT_DESCRIBE = re.compile(
    r"^(?P<tag>.+)-(?P<distance>\d+)-g(?P<commit>[0-9a-f]+)(?P<dirty>-dirty)?$"
)

#: Matches the trailing numeric component of a release tag.
_TRAILING_NUMBER = re.compile(r"^(?P<head>.*?)(?P<last>\d+)$")


def _to_pep440(version: str) -> str | None:
    """Turn a ``git describe`` string into a PEP 440 version.

    ``git describe`` renders "3 commits past tag 2026.6.0" as
    ``2026.6.0-3-gc98fef9``, which is not a valid PEP 440 version: passing it
    to ``setup()`` fails outright. Anything else is returned untouched.

    The result mirrors what ``setuptools_scm`` would produce for the same
    commit under the ``guess-next-dev`` scheme, so that building from a source
    tarball and building from a clone of the same commit agree.

    Args:
        version: Version string as read from its source.

    Returns:
        A PEP 440 compliant version, or ``None`` when the string describes a
        tag that cannot be expressed as one, so the caller can move on to the
        next source rather than emit something invalid.

    """
    match = _GIT_DESCRIBE.match(version.strip())
    if match is None:
        # An exact tag, or a version that already went through setuptools_scm.
        return version

    # guess-next-dev counts as a development release of the *next* version, so
    # bump the trailing number of the tag: 2026.6.0 + 3 commits -> 2026.6.1.dev3
    bumped = _TRAILING_NUMBER.match(match.group("tag"))
    if bumped is None:
        # A tag with no trailing number (say "nightly") cannot be bumped, and
        # neither ".postN" nor any other suffix would make it PEP 440 valid.
        return None
    distance = match.group("distance")
    head, last = bumped.group("head"), int(bumped.group("last"))
    return f"{head}{last + 1}.dev{distance}"


def _from_environment(root: pathlib.Path) -> str | None:
    """Read an explicitly pinned version from the environment."""
    del root  # Unused: kept so every reader shares one signature.
    value = os.environ.get("PYINTERP_VERSION", "").strip()
    return value or None


def has_setuptools_scm() -> bool:
    """Return whether ``setuptools_scm`` can be imported."""
    try:
        import setuptools_scm  # noqa: F401
    except ImportError:
        return False
    return True


def _from_git(root: pathlib.Path) -> str | None:
    """Ask setuptools_scm for the version described by the git checkout."""
    if not (root / ".git").exists():
        return None
    try:
        import setuptools_scm
    except ImportError:
        return None
    try:
        return setuptools_scm.get_version(
            root=str(root),
            version_scheme="guess-next-dev",
            local_scheme="no-local-version",
        )
    except (LookupError, OSError):
        # No tags, a shallow clone, or no usable git binary. Not fatal: the
        # next source takes over.
        return None


def _from_version_file(root: pathlib.Path) -> str | None:
    """Read VERSION.txt, once ``git archive`` has expanded it."""
    path = root / "VERSION.txt"
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8").strip()
    # Still the raw placeholder: this is a checkout, not an exported tarball.
    if not content or content.startswith("$Format:"):
        return None
    # The file holds raw `git describe` output, which is not PEP 440.
    return _to_pep440(content)


def _from_python_module(root: pathlib.Path) -> str | None:
    """Read the ``_version.py`` module generated by setuptools_scm."""
    path = root / "pyinterp" / "_version.py"
    if not path.is_file():
        return None
    match = _VERSION_ASSIGNMENT.search(path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _from_pkg_info(root: pathlib.Path) -> str | None:
    """Read the ``Version:`` field of an unpacked sdist."""
    path = root / "PKG-INFO"
    if not path.is_file():
        return None
    match = _PKG_INFO_VERSION.search(path.read_text(encoding="utf-8"))
    if match is None:
        return None
    return match.group(1).strip() or None


#: Resolution chain, in precedence order. Each entry pairs the label reported
#: back to the caller with the reader implementing it.
_SOURCES: tuple[tuple[str, Callable[[pathlib.Path], str | None]], ...] = (
    ("environment", _from_environment),
    ("git", _from_git),
    ("VERSION.txt", _from_version_file),
    ("pyinterp/_version.py", _from_python_module),
    ("PKG-INFO", _from_pkg_info),
)


def resolve_version(root: pathlib.Path) -> tuple[str, str]:
    """Resolve the project version.

    Args:
        root: Root of the pyinterp source tree.

    Returns:
        The version and the name of the source it was read from. The source is
        ``"fallback"`` when nothing could be determined, in which case the
        version is :data:`FALLBACK_VERSION` and is not meaningful.

    """
    for name, reader in _SOURCES:
        version = reader(root)
        if version:
            return version, name
    return FALLBACK_VERSION, "fallback"


def write_version_module(root: pathlib.Path, version: str) -> None:
    """Write ``pyinterp/_version.py`` so the installed package reports it.

    ``setuptools_scm`` writes this module itself when it drives the build. This
    fills it in for the builds it cannot serve -- a source tarball, or an
    environment without ``setuptools_scm`` -- because ``pyinterp/__init__.py``
    reads its version from there.

    Args:
        root: Root of the pyinterp source tree.
        version: Version to record.

    """
    # Keep the leading numeric components: "2026.6.1.dev3" -> (2026, 6, 1).
    parts: list[int] = []
    for component in version.replace("-", ".").split("."):
        if not component.isdigit():
            break
        parts.append(int(component))
    # A tuple literal needs a trailing comma to stay a tuple when it holds a
    # single element, but "(,)" is a syntax error, so the empty case -- a
    # version with no leading digit at all -- is spelled out separately.
    version_tuple = (
        "(" + "".join(f"{item}, " for item in parts).rstrip() + ")"
        if parts
        else "()"
    )

    path = root / "pyinterp" / "_version.py"
    path.write_text(
        "# file generated by the pyinterp build\n"
        "# don't change, don't track in version control\n"
        "\n"
        '__all__ = ["__version__", "__version_tuple__", "version", '
        '"version_tuple"]\n'
        "\n"
        f"__version__ = version = {version!r}\n"
        f"__version_tuple__ = version_tuple = {version_tuple}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).parent.parent.absolute()
    resolved, origin = resolve_version(root_dir)
    print(f"{resolved} (from {origin})")
