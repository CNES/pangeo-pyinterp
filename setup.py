# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Setup script for the pyinterp package."""

from __future__ import annotations

import os
import pathlib
import platform
import sys
import sysconfig
from typing import ClassVar

import setuptools
import setuptools.command.build_ext


# Type alias for user options
USER_OPTIONS = list[tuple[str, str, str]] | list[tuple[str, str | None, str]]

# Check Python requirement
MAJOR = sys.version_info[0]
MINOR = sys.version_info[1]

# Working directory
WORKING_DIRECTORY = pathlib.Path(__file__).parent.absolute()

# OSX deployment target
OSX_DEPLOYMENT_TARGET = "13.3"

# MKL-FFT library
MKL_FFT = "mkl"

# PocketFFT library
POCKETFFT = "pocketfft"

# FFT choices
FFT_CHOICES = [MKL_FFT, POCKETFFT]


def compare_setuptools_version(required: tuple[int, ...]) -> bool:
    """Compare the version of setuptools with the required version."""
    current = tuple(map(int, setuptools.__version__.split(".")[:2]))
    return current >= required


def distutils_dirname(
    prefix: str | None = None,
    extname: str | None = None,
) -> pathlib.Path:
    """Return the name of the build directory."""
    prefix = prefix or "lib"
    extname = "" if extname is None else os.sep.join(extname.split(".")[:-1])
    if compare_setuptools_version((62, 1)):
        return pathlib.Path(
            WORKING_DIRECTORY,
            "build",
            f"{prefix}.{sysconfig.get_platform()}-"
            f"{sys.implementation.cache_tag}",
            extname,
        )
    return pathlib.Path(
        WORKING_DIRECTORY,
        "build",
        f"{prefix}.{sysconfig.get_platform()}-{MAJOR}.{MINOR}",
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
) -> None:
    """Update cmake and build arguments based on the platform."""
    # Calculate memory-aware parallel jobs
    parallel_jobs = get_parallel_jobs()

    if not is_windows:
        build_args += ["--", f"-j{parallel_jobs}"]
        if platform.system() == "Darwin":
            cmake_args += [
                f"-DCMAKE_OSX_DEPLOYMENT_TARGET={OSX_DEPLOYMENT_TARGET}"
            ]
    else:
        cmake_args += [
            "-DCMAKE_GENERATOR_PLATFORM=x64",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={extdir}",
        ]
        build_args += ["--", f"/m:{parallel_jobs}"]


# pylint: disable=too-many-instance-attributes
class BuildExt(setuptools.command.build_ext.build_ext):
    """Build everything needed to install."""

    user_options = setuptools.command.build_ext.build_ext.user_options
    user_options += [
        (
            "build-unittests",
            None,
            "Build the unit tests of the C++ extension",
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

    boolean_options = setuptools.command.build_ext.build_ext.boolean_options
    boolean_options += [
        "enable_coverage",
        "export_compile_commands",
        "mkl",
        "reconfigure",
    ]

    def initialize_options(self) -> None:
        """Set the default values of the options."""
        super().initialize_options()
        self.export_compile_commands = None
        self.enable_coverage = None
        self.c_compiler = None
        self.cxx_compiler = None
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

        if conda_prefix is not None:
            result.append("-DCMAKE_PREFIX_PATH=" + conda_prefix)

        if self.mkl or self.fft == MKL_FFT:
            self.set_conda_mklroot()

        return result

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
            return pathlib.Path(self.build_lib).joinpath(
                *ext.name.split(".")[:-1]
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
        cmake_args: list[str] = [
            "-DCMAKE_BUILD_TYPE=" + cfg,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPython3_EXECUTABLE=" + sys.executable,
            *self.set_cmake_user_options(),
        ]
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
        if not self.dry_run:
            cmake_cmd = ["cmake", "--build", ".", "--target", "core"]
            self.spawn(cmake_cmd + build_args)  # type: ignore[arg-type]
        os.chdir(str(WORKING_DIRECTORY))

    # pylint: enable=too-many-instance-attributes


class CxxTestRunner(setuptools.Command):
    """Compile and launch the C++ tests."""

    description: ClassVar[str] = "run the C++ tests"
    user_options: ClassVar[USER_OPTIONS] = []

    def initialize_options(self) -> None:
        """Set the default values of the options."""
        if platform.system() == "Windows":
            raise RuntimeError("Code coverage is not supported on Windows")

    def finalize_options(self) -> None:
        """Set final values for all the options that this command supports."""

    def run(self) -> None:
        """Run tests."""
        # Directory used during the generating the C++ extension.
        tempdir = distutils_dirname("temp")

        # Navigate to the directory containing the C++ tests and run them.
        os.chdir(str(tempdir / "cxx" / "tests"))
        self.spawn(["ctest", "--output-on-failure"])

        # File containing the coverage report.
        coverage_lcov = str(
            pathlib.Path(tempdir.parent.parent, "coverage_cpp.lcov")
        )

        # Collect coverage data from python/C++ unit tests
        self.spawn(
            [
                "lcov",
                "--capture",
                "--directory",
                str(tempdir),
                "--output-file",
                coverage_lcov,
            ]
        )


def long_description() -> str:
    """Read the README file."""
    with pathlib.Path(WORKING_DIRECTORY, "README.rst").open(
        encoding="utf-8"
    ) as stream:
        return stream.read()


def main() -> None:
    """Set up module."""
    install_requires = ["dask", "numpy", "xarray >= 0.13"]
    setuptools.setup(
        author="CNES/CLS",
        author_email="fbriol@gmail.com",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Topic :: Scientific/Engineering :: Physics",
            "Natural Language :: English",
            "Operating System :: POSIX",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
        ],
        cmdclass={
            "build_ext": BuildExt,
            "gtest": CxxTestRunner,
        },
        description="Interpolation of geo-referenced data for Python.",
        ext_modules=[CMakeExtension(name="pyinterp.core")],
        install_requires=install_requires,
        include_package_data=True,
        keywords="interpolation, geospatial, geohash, geodetic",
        license="BSD-3-Clause",
        license_files=("LICENSE",),
        long_description=long_description(),
        long_description_content_type="text/x-rst",
        name="pyinterp",
        package_data={
            "pyinterp": ["py.typed", "*.pyi"],
            "pyinterp.tests": ["dataset/*"],
        },
        package_dir={"pyinterp": "pyinterp"},
        packages=setuptools.find_namespace_packages(
            include=["pyinterp", "pyinterp.*"],
        ),
        platforms=["POSIX", "MacOS", "Windows"],
        python_requires=">=3.11",
        url="https://github.com/CNES/pangeo-pyinterp",
        zip_safe=False,
    )


if __name__ == "__main__":
    if platform.system() == "Darwin":
        os.environ["MACOSX_DEPLOYMENT_TARGET"] = OSX_DEPLOYMENT_TARGET
    main()
