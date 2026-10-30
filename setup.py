# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Setup script for the pyinterp package."""

from __future__ import annotations

import functools
import os
import pathlib
import platform
import re
import shlex
import sys
import sysconfig
from typing import ClassVar

import setuptools
import setuptools.command.bdist_wheel
import setuptools.command.build_ext
import setuptools.command.build_py


# Import the geometry stub generator and the version resolver from scripts/ at
# module load time so the commands below can use them without an in-function
# import.
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute() / "scripts"))
import generate_geometry_stubs  # type: ignore[import-not-found]
import resolve_version  # type: ignore[import-not-found]


sys.path.pop(0)


# Working directory
WORKING_DIRECTORY = pathlib.Path(__file__).parent.absolute()

# OSX deployment target
OSX_DEPLOYMENT_TARGET = "13.4"

# MKL-FFT library
MKL_FFT = "mkl"

# PocketFFT library
POCKETFFT = "pocketfft"

# FFT choices
FFT_CHOICES = [MKL_FFT, POCKETFFT]

# Project version and the source it was read from. Resolved once here so that
# the metadata, the generated _version.py and the CMake configuration all agree.
VERSION, VERSION_SOURCE = resolve_version.resolve_version(WORKING_DIRECTORY)

# nanobind backend module the extension is built against in "split mode" (see
# PYINTERP_NANOBIND_BACKEND_MODULE in CMakeLists.txt). Only the binary wheels
# set it: source builds keep linking their own copy of nanobind.
NANOBIND_BACKEND_MODULE = os.environ.get(
    "PYINTERP_NANOBIND_BACKEND_MODULE", ""
)

# Oldest CPython supported by pyinterp: the floor of the stable ABI targeted by
# the split-mode wheels (cp312-abi3).
MIN_PYTHON_VERSION = (3, 12)

# Free-threaded CPython only has a stable ABI ("abi3t", PEP 803) from 3.15.
MIN_ABI3T_PYTHON_VERSION = (3, 15)


def is_free_threaded() -> bool:
    """Return whether the running interpreter is a free-threaded build."""
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


@functools.cache
def use_split_mode() -> bool:
    """Return whether the extension is built in nanobind split mode.

    Split mode always targets a stable ABI, which free-threaded Python lacks
    before 3.15: these interpreters fall back to the linked mode, and get a
    wheel tied to their version.
    """
    if not NANOBIND_BACKEND_MODULE:
        return False
    if is_free_threaded() and sys.version_info < MIN_ABI3T_PYTHON_VERSION:
        print(
            "WARNING: nanobind split mode requires the abi3t stable ABI of "
            "Python 3.15 or newer on free-threaded interpreters; building "
            f"Python {sys.version_info.major}.{sys.version_info.minor}t in "
            "linked mode.",
            file=sys.stderr,
        )
        return False
    return True


def nanobind_backend_requirement() -> str | None:
    """Return the requirement on the backend the extension is built against.

    Only the ``nanobind_backend`` module has a known PyPI package; the floor is
    the backend ABI version of the vendored nanobind (no upper bound, newer
    backends serve older extensions).
    """
    if NANOBIND_BACKEND_MODULE != "nanobind_backend":
        return None
    header = pathlib.Path(
        WORKING_DIRECTORY,
        "third_party",
        "nanobind",
        "include",
        "nanobind",
        "nb_backend.h",
    ).read_text(encoding="utf-8")
    version = dict(
        re.findall(
            r"^#define NB_BACKEND_ABI_(MAJOR|MINOR) (\d+)", header, re.M
        )
    )
    return f"nanobind-backend>={version['MAJOR']}.{version['MINOR']}"


def distutils_dirname(
    prefix: str | None = None,
    extname: str | None = None,
) -> pathlib.Path:
    """Return the name of the build directory."""
    prefix = prefix or "lib"
    extname = "" if extname is None else os.sep.join(extname.split(".")[:-1])
    return pathlib.Path(
        WORKING_DIRECTORY,
        "build",
        f"{prefix}.{sysconfig.get_platform()}-{sys.implementation.cache_tag}",
        extname,
    )


# pylint: disable=too-few-public-methods
class CMakeExtension(setuptools.Extension):
    """Python extension to build."""

    def __init__(self, name: str) -> None:
        """Initialize the extension."""
        super().__init__(name, sources=[])

    # pylint: enable=too-few-public-methods


def get_parallel_jobs() -> int:
    """Calculate optimal number of parallel jobs based on available memory.

    C++ compilation is memory-intensive. Each compilation unit can use
    500MB-2GB depending on template usage. This function estimates a safe
    parallelism level to avoid memory exhaustion.

    Users can override this with the CMAKE_BUILD_PARALLEL_LEVEL environment
    variable.

    Returns:
        Number of parallel jobs to use

    """
    # Allow user override via environment variable
    if "CMAKE_BUILD_PARALLEL_LEVEL" in os.environ:
        try:
            user_jobs = int(os.environ["CMAKE_BUILD_PARALLEL_LEVEL"])
            if user_jobs > 0:
                return user_jobs
        except ValueError:
            pass

    return os.cpu_count() or 4


def prepare_cmake_arguments(
    is_windows: bool,
    config: str,
    extdir: str,
    cmake_args: list[str],
    build_args: list[str],
    generator: str | None = None,
) -> None:
    """Update cmake and build arguments based on the platform."""
    # Calculate memory-aware parallel jobs
    parallel_jobs = get_parallel_jobs()

    # The multi-config Visual Studio generators need the platform flag and
    # MSBuild-style parallel switch; single-config generators (e.g. Ninja) do
    # not understand them and rely on the active toolchain for the target
    # architecture instead.
    is_msbuild = generator is None or generator.startswith("Visual Studio")

    if not is_windows:
        build_args += ["--", f"-j{parallel_jobs}"]
        if platform.system() == "Darwin":
            cmake_args += [
                f"-DCMAKE_OSX_DEPLOYMENT_TARGET={OSX_DEPLOYMENT_TARGET}"
            ]
    elif is_msbuild:
        cmake_args += [
            "-DCMAKE_GENERATOR_PLATFORM=x64",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={extdir}",
        ]
        build_args += ["--", f"/m:{parallel_jobs}"]
    else:
        build_args += ["--", f"-j{parallel_jobs}"]


# pylint: disable=too-many-instance-attributes
class BuildExt(setuptools.command.build_ext.build_ext):
    """Build everything needed to install."""

    user_options: ClassVar[list[tuple[str, str | None, str]]] = [
        *setuptools.command.build_ext.build_ext.user_options,
        (
            "build-unittests",
            None,
            "Build the C++ unit tests in addition to the extension",
        ),
        (
            "enable-coverage",
            None,
            "Enable coverage reporting",
        ),
        (
            "export-compile-commands",
            None,
            "Export compile commands for tooling",
        ),
        (
            "fft=",
            None,
            "Select FFT library: pocketfft or MKL",
        ),
        (
            "cmake-args=",
            None,
            "Additional arguments for CMake",
        ),
        (
            "cxx-compiler=",
            None,
            "Preferred C++ compiler",
        ),
        (
            "generator=",
            None,
            "Selected CMake generator",
        ),
        (
            "mkl=",
            None,
            "Using MKL as BLAS library",
        ),
        (
            "reconfigure",
            None,
            "Forces CMake to reconfigure this project",
        ),
    ]

    boolean_options: ClassVar[list[str]] = [
        *setuptools.command.build_ext.build_ext.boolean_options,
        "build_unittests",
        "enable_coverage",
        "export_compile_commands",
        "mkl",
        "reconfigure",
    ]

    def initialize_options(self) -> None:
        """Set the default values of the options."""
        super().initialize_options()
        self.build_unittests = None
        self.cmake_args = None
        self.cxx_compiler = None
        self.enable_coverage = None
        self.export_compile_commands = None
        self.fft = None
        self.generator = None
        self.mkl = None
        self.reconfigure = None

    def finalize_options(self) -> None:
        """Set final values for all the options that this command supports."""
        super().finalize_options()
        if self.enable_coverage is not None and platform.system() == "Windows":
            raise RuntimeError("Code coverage is not supported on Windows")
        if self.fft is not None and self.fft not in FFT_CHOICES:
            raise ValueError(
                f"Invalid value for option 'fft'. "
                f"Choose one of: {', '.join(FFT_CHOICES)}",
            )

    def run(self) -> None:
        """Carry out the action."""
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_extension(self, ext: setuptools.Extension) -> None:
        """Build a single extension.

        For CMakeExtension, skip the default build process since cmake already
        built it.
        """
        if isinstance(ext, CMakeExtension):
            return
        super().build_extension(ext)

    def get_outputs(self) -> list[str]:
        """Return the list of files generated by building."""
        return super().get_outputs()

    @staticmethod
    def set_conda_mklroot() -> None:
        """Set the default MKL path in Anaconda's environment."""
        mkl_header = pathlib.Path(sys.prefix, "include", "mkl.h")
        if not mkl_header.exists():
            mkl_header = pathlib.Path(
                sys.prefix, "Library", "include", "mkl.h"
            )

        if mkl_header.exists():
            os.environ["MKLROOT"] = sys.prefix

    @staticmethod
    def conda_prefix() -> str | None:
        """Return the conda prefix."""
        if "CONDA_PREFIX" in os.environ:
            return os.environ["CONDA_PREFIX"]
        return None

    @staticmethod
    def split_cmake_args(value: str | None) -> list[str]:
        """Split CMake arguments preserving quoted values."""
        if not value:
            return []
        lexer = shlex.shlex(value, posix=True)
        lexer.whitespace_split = True
        if os.name == "nt":
            # The backslashes of Windows paths are separators, not escapes.
            lexer.escape = ""
        return list(lexer)

    @staticmethod
    def filter_cmake_definitions(args: list[str], *names: str) -> list[str]:
        """Filter out ``-D<name>=...`` CMake cache definitions."""
        prefixes = tuple(f"-D{name}=" for name in names)
        return [arg for arg in args if not arg.startswith(prefixes)]

    @staticmethod
    def has_cmake_prefix_path(args: list[str]) -> bool:
        """Return whether CMake args already define CMAKE_PREFIX_PATH."""
        return any(arg.startswith("-DCMAKE_PREFIX_PATH=") for arg in args)

    @staticmethod
    def preferred_cmake_prefix_path(conda_prefix: str | None) -> str | None:
        """Get preferred CMAKE_PREFIX_PATH from environment or conda."""
        return os.environ.get("CMAKE_PREFIX_PATH") or conda_prefix

    def set_cmake_user_options(self) -> list[str]:
        """Set the options defined by the user."""
        result = []

        conda_prefix = self.conda_prefix()

        if self.cxx_compiler is not None:
            result.append("-DCMAKE_CXX_COMPILER=" + self.cxx_compiler)

        if self.enable_coverage:
            result.append("-DENABLE_COVERAGE=ON")

        if self.export_compile_commands:
            result.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")

        if self.fft is not None:
            result.append(f"-DFFT_IMPLEMENTATION={self.fft}")

        # Preserve cross-compilation/toolchain arguments injected by
        # conda-forge (e.g. CMAKE_SYSTEM_PROCESSOR, sysroot, emulator).
        result.extend(self.split_cmake_args(os.environ.get("CMAKE_ARGS")))

        result.extend(self.split_cmake_args(self.cmake_args))

        # Keep host-interpreter execution stable for build-time tools
        # (e.g. nanobind stub generation).
        result = self.filter_cmake_definitions(result, "Python_EXECUTABLE")

        if not self.has_cmake_prefix_path(result):
            prefix_path = self.preferred_cmake_prefix_path(conda_prefix)
            if prefix_path:
                result.append(f"-DCMAKE_PREFIX_PATH={prefix_path}")

        if self.mkl or self.fft == MKL_FFT:
            self.set_conda_mklroot()

        return result

    @staticmethod
    def is_cross_compiling() -> bool:
        """Return whether conda build cross-compilation is active."""
        return os.environ.get("CONDA_BUILD_CROSS_COMPILATION") == "1"

    @staticmethod
    def host_python_executable() -> str:
        """Return a host-runnable Python executable for build-time tooling."""
        candidates = [
            os.environ.get("PYTHON_FOR_BUILD"),
            (
                str(pathlib.Path(os.environ["BUILD_PREFIX"], "bin", "python"))
                if "BUILD_PREFIX" in os.environ
                else None
            ),
            sys.executable,
        ]
        for item in candidates:
            if item and pathlib.Path(item).exists():
                return item
        return sys.executable

    def get_config(self) -> str:
        """Return the configuration to use."""
        cfg: str
        if self.debug:
            cfg = "Debug"
        elif self.enable_coverage:
            cfg = "RelWithDebInfo"
        else:
            cfg = "Release"
        return cfg

    def get_extdir(self, ext: CMakeExtension) -> pathlib.Path:
        """Detect if the build is in editable mode."""
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        # If the extension is built in the "root/build/lib.*" directory,
        # then it is not an editable install.
        if distutils_dirname().resolve() != extdir.parent:
            return (
                pathlib.Path(self.build_lib)
                .resolve()
                .joinpath(*ext.name.split(".")[:-1])
            )
        return extdir

    def build_cmake(self, ext: CMakeExtension) -> None:
        """Execute cmake to build the Python extension."""
        # These dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(WORKING_DIRECTORY, self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = str(self.get_extdir(ext))

        cfg = self.get_config()
        is_cross = self.is_cross_compiling()
        host_python = self.host_python_executable()
        cmake_args: list[str] = [
            "-DCMAKE_BUILD_TYPE=" + cfg,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            # Hand CMake the version resolved here rather than letting it run
            # its own detection: the extension must report the same version as
            # the Python package that ships it.
            "-DPYINTERP_VERSION=" + VERSION,
            # Always set, so that reconfiguring a build tree switches the mode.
            "-DPYINTERP_NANOBIND_BACKEND_MODULE="
            + (NANOBIND_BACKEND_MODULE if use_split_mode() else ""),
            *self.set_cmake_user_options(),
        ]
        if is_cross:
            # The host interpreter cannot be discovered by CMake when
            # cross-compiling (it would pick up the target arch). Pass it
            # explicitly; CMakeLists.txt reads PYINTERP_HOST_PYTHON_EXECUTABLE
            # inside its cross-compile branch.
            cmake_args.append(
                "-DPYINTERP_HOST_PYTHON_EXECUTABLE=" + host_python
            )
            prefix = os.environ.get("PREFIX")
            if prefix:
                cmake_args.append("-DPython_ROOT_DIR=" + prefix)
        else:
            cmake_args.append("-DPython_EXECUTABLE=" + host_python)
        build_args = ["--config", cfg]

        is_windows = platform.system() == "Windows"

        # Determine the generator to use
        generator = None
        if self.generator is not None:
            generator = self.generator
            cmake_args.append("-G" + generator)
        elif is_windows:
            if "CMAKE_GEN" in os.environ:
                generator = os.environ["CMAKE_GEN"]
            else:
                generator = "Visual Studio 17 2022"
            cmake_args.append("-G" + generator)

        prepare_cmake_arguments(
            is_windows,
            cfg,
            extdir,
            cmake_args,
            build_args,
            generator,
        )

        os.chdir(str(build_temp))

        # Configure CMake if needed or requested
        configure = (
            (self.reconfigure is not None)
            if pathlib.Path(
                build_temp,
                "CMakeFiles",
                "TargetDirectories.txt",
            ).exists()
            else True
        )

        if configure:
            self.spawn(["cmake", str(WORKING_DIRECTORY), *cmake_args])
        # The default target also builds the C++ unit tests, when GoogleTest
        # is found.
        cmake_cmd = ["cmake", "--build", "."]
        if not self.build_unittests:
            cmake_cmd += ["--target", "core"]
        self.spawn(cmake_cmd + build_args)  # type: ignore[arg-type]
        os.chdir(str(WORKING_DIRECTORY))

    # pylint: enable=too-many-instance-attributes


class BdistWheel(setuptools.command.bdist_wheel.bdist_wheel):
    """Tag the split-mode wheels with the stable ABI of their extension.

    The extension is built by CMake, so setuptools cannot tell that it targets
    the stable ABI. Its own ``py_limited_api`` option is not used either: on a
    free-threaded interpreter it writes ``abi3.abi3t``, which also matches the
    GIL-enabled interpreters, whereas the free-threaded frontend needs a
    free-threaded backend and only loads there.
    """

    def get_tag(self) -> tuple[str, str, str]:
        """Return the (python, abi, platform) tag of the wheel."""
        impl, abi, plat = super().get_tag()
        if not use_split_mode():
            return impl, abi, plat
        # The floors match the Py_LIMITED_API targeted by cxx/CMakeLists.txt.
        if is_free_threaded():
            return "cp{}{}".format(*MIN_ABI3T_PYTHON_VERSION), "abi3t", plat
        return "cp{}{}".format(*MIN_PYTHON_VERSION), "abi3", plat


class BuildPy(setuptools.command.build_py.build_py):
    """Regenerate the ``pyinterp.geometry`` stub tree before packaging.

    The shims under ``pyinterp/geometry/`` mirror ``pyinterp/core/geometry/``
    so that IDEs and type checkers can resolve the import path that
    ``pyinterp/__init__.py`` registers dynamically via ``sys.modules``.
    Regenerating here means sdists and wheels always ship a tree in sync
    with the core stubs, even if a contributor forgot to run the script.
    """

    def run(self) -> None:
        """Regenerate the geometry stub tree, then run the normal build."""
        generate_geometry_stubs.generate()
        super().run()


def long_description() -> str:
    """Read the README file."""
    with pathlib.Path(WORKING_DIRECTORY, "README.rst").open(
        encoding="utf-8"
    ) as stream:
        return stream.read()


def version_keywords() -> dict[str, object]:
    """Return the version-related keyword arguments for ``setup()``.

    ``setuptools_scm`` stays in charge of the packaging side whenever it is
    available: it writes ``pyinterp/_version.py`` and stamps the sdist
    ``PKG-INFO``. Declaring ``use_scm_version`` explicitly is required because
    the ``[tool.setuptools_scm]`` table alone does not trigger the hook (there
    is no ``[project]`` table declaring a dynamic version).

    Outside a git checkout it cannot discover a version on its own and aborts
    the build with ``LookupError``. ``SETUPTOOLS_SCM_PRETEND_VERSION_FOR_*`` is
    its supported way of being handed one, so the version resolved from the
    other sources is passed through it -- which keeps a single code path and
    avoids the "version already set" warning that setting ``version=`` on top
    of it would raise.
    """
    if VERSION_SOURCE == "fallback":
        print(
            "WARNING: could not determine the pyinterp version (no git "
            "checkout, VERSION.txt, pyinterp/_version.py or PKG-INFO found). "
            f"Building as {VERSION}. Set the PYINTERP_VERSION environment "
            "variable to pin it explicitly.",
            file=sys.stderr,
        )

    if resolve_version.has_setuptools_scm():
        # The sdist ships pyinterp/_version.py (MANIFEST.in), so it has to be
        # written to the source tree. setuptools_scm >= 10.1 warns that this
        # default will change; the variable is ignored by older releases.
        os.environ.setdefault("SETUPTOOLS_SCM_WRITE_TO_SOURCE", "1")
        if VERSION_SOURCE != "git":
            os.environ.setdefault(
                "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYINTERP", VERSION
            )
        return {
            "use_scm_version": {
                "version_file": "pyinterp/_version.py",
                "version_scheme": "guess-next-dev",
                "local_scheme": "no-local-version",
            }
        }

    # No setuptools_scm at all: do its job by hand. pyinterp/__init__.py reads
    # its version from _version.py at import time, so the module has to exist.
    resolve_version.write_version_module(WORKING_DIRECTORY, VERSION)
    return {"version": VERSION}


def main() -> None:
    """Set up module."""
    install_requires = ["numpy"]
    # Split-mode wheels need the backend at runtime, source builds do not.
    backend = nanobind_backend_requirement() if use_split_mode() else None
    if backend is not None:
        install_requires.append(backend)
    setuptools.setup(
        author="CNES/CLS",
        author_email="fbriol@gmail.com",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Atmospheric Science",
            "Topic :: Scientific/Engineering :: GIS",
            "Topic :: Scientific/Engineering :: Hydrology",
            "Topic :: Scientific/Engineering :: Oceanography",
            "Topic :: Scientific/Engineering :: Physics",
            "Natural Language :: English",
            "Operating System :: POSIX",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Programming Language :: Python :: 3.14",
            "Programming Language :: Python :: 3.15",
        ],
        cmdclass={
            "bdist_wheel": BdistWheel,
            "build_ext": BuildExt,
            "build_py": BuildPy,
        },
        description=(
            "Fast interpolation of geo-referenced data for Python and xarray."
        ),
        ext_modules=[CMakeExtension(name="pyinterp.core")],
        extras_require={
            "dask": ["dask[array]"],
            "numba": ["numba"],
            "xarray": ["xarray >= 0.13"],
        },
        install_requires=install_requires,
        include_package_data=True,
        keywords=(
            "interpolation, xarray, regridding, geospatial, geodetic, "
            "bilinear, bicubic, spline, kriging, rtree, oceanography, "
            "climate, satellite, netcdf, geohash"
        ),
        license="BSD-3-Clause",
        license_files=("LICENSE",),
        long_description=long_description(),
        long_description_content_type="text/x-rst",
        name="pyinterp",
        package_data={
            "pyinterp": ["py.typed", "*.pyi"],
            "pyinterp.geometry": ["*.pyi"],
            "pyinterp.geometry.cartesian": ["*.pyi"],
            "pyinterp.geometry.geographic": ["*.pyi"],
            "pyinterp.tests": ["dataset/*"],
        },
        package_dir={"pyinterp": "pyinterp"},
        packages=setuptools.find_namespace_packages(
            include=["pyinterp", "pyinterp.*"],
        ),
        project_urls={
            "Documentation": "https://cnes.github.io/pangeo-pyinterp/",
            "Changelog": (
                "https://cnes.github.io/pangeo-pyinterp/changelog.html"
            ),
            "Source": "https://github.com/CNES/pangeo-pyinterp",
            "Issues": "https://github.com/CNES/pangeo-pyinterp/issues",
        },
        python_requires=">={}.{}".format(*MIN_PYTHON_VERSION),
        url="https://github.com/CNES/pangeo-pyinterp",
        **version_keywords(),  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    if platform.system() == "Darwin":
        os.environ["MACOSX_DEPLOYMENT_TARGET"] = OSX_DEPLOYMENT_TARGET
    main()
