# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""This script is the entry point for building, distributing and installing
this module using distutils/setuptools."""
from __future__ import annotations

from typing import Any, ClassVar
import datetime
import os
import pathlib
import platform
import re
import subprocess
import sys
import sysconfig

import setuptools
import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.sdist
import setuptools.command.test

# Check Python requirement
MAJOR = sys.version_info[0]
MINOR = sys.version_info[1]

# Working directory
WORKING_DIRECTORY = pathlib.Path(__file__).parent.absolute()

# OSX deployment target
OSX_DEPLOYMENT_TARGET = '11.0'


def compare_setuptools_version(required: tuple[int, ...]) -> bool:
    """Compare the version of setuptools with the required version."""
    current = tuple(map(int, setuptools.__version__.split('.')[:2]))
    return current >= required


def distutils_dirname(prefix=None, extname=None) -> pathlib.Path:
    """Returns the name of the build directory."""
    prefix = prefix or 'lib'
    extname = '' if extname is None else os.sep.join(extname.split('.')[:-1])
    if compare_setuptools_version((62, 1)):
        return pathlib.Path(
            WORKING_DIRECTORY, 'build', f'{prefix}.{sysconfig.get_platform()}-'
            f'{sys.implementation.cache_tag}', extname)
    return pathlib.Path(
        WORKING_DIRECTORY, 'build',
        f'{prefix}.{sysconfig.get_platform()}-{MAJOR}.{MINOR}', extname)


def execute(cmd) -> str:
    """Executes a command and returns the lines displayed on the standard
    output."""
    with subprocess.Popen(cmd,
                          shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as process:
        assert process.stdout is not None
        return process.stdout.read().decode()


def update_meta(path, version) -> None:
    """Updating the version number description in conda/meta.yaml."""
    with open(path, encoding='utf-8') as stream:
        lines = stream.readlines()
    pattern = re.compile(r'{% set version = ".*" %}')

    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if match is not None:
            lines[idx] = f'{{% set version = "{version}" %}}\n'

    with open(path, 'w', encoding='utf-8') as stream:
        stream.write(''.join(lines))


def update_environment(path, version) -> None:
    """Updating the version number description in conda environment."""
    with open(path, encoding='utf-8') as stream:
        lines = stream.readlines()
    pattern = re.compile(r'(\s+-\s+pyinterp)\s*>=\s*(.+)')

    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if match is not None:
            lines[idx] = f'{match.group(1)}>={version}\n'

    with open(path, 'w', encoding='utf-8') as stream:
        stream.write(''.join(lines))


def revision() -> str:
    """Returns the software version."""
    os.chdir(WORKING_DIRECTORY)
    module = pathlib.Path(WORKING_DIRECTORY, 'src', 'pyinterp', 'version.py')

    # If the ".git" directory exists, this function is executed in the
    # development environment, otherwise it's a release.
    if not pathlib.Path(WORKING_DIRECTORY, '.git').exists():
        pattern = re.compile(r'return "(\d+\.\d+\.\d+)"')
        with open(module, encoding='utf-8') as stream:
            for line in stream:
                match = pattern.search(line)
                if match:
                    return match.group(1)
        raise AssertionError

    stdout: Any = execute(
        'git describe --tags --dirty --long --always').strip()
    pattern = re.compile(r'([\w\d\.]+)-(\d+)-g([\w\d]+)(?:-(dirty))?')
    match = pattern.search(stdout)
    if match is None:
        # No tag found, use the last commit
        pattern = re.compile(r'([\w\d]+)(?:-(dirty))?')
        match = pattern.search(stdout)
        assert match is not None, f'Unable to parse git output {stdout!r}'
        version = '0.0'
        sha1 = match.group(1)
    else:
        version = match.group(1)
        commits = int(match.group(2))
        sha1 = match.group(3)
        if commits != 0:
            version += f'.dev{commits}'

    stdout = execute(f"git log  {sha1} -1 --format=\"%H %at\"")
    stdout = stdout.strip().split()
    date = datetime.datetime.fromtimestamp(int(stdout[1]))

    # Conda configuration files are not present in the distribution, but only
    # in the GIT repository of the source code.
    meta = pathlib.Path(WORKING_DIRECTORY, 'conda', 'meta.yaml')
    if meta.exists():
        update_meta(meta, version)
        update_environment(
            pathlib.Path(WORKING_DIRECTORY, 'conda', 'environment.yml'),
            version)
        update_environment(
            pathlib.Path(WORKING_DIRECTORY, 'binder', 'environment.yml'),
            version)

    # Updating the version number description for sphinx
    conf = pathlib.Path(WORKING_DIRECTORY, 'docs', 'source', 'conf.py')
    with open(conf, encoding='utf-8') as stream:
        lines = stream.readlines()
    pattern = re.compile(r'(\w+)\s+=\s+(.*)')

    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if match is not None:
            if match.group(1) == 'version':
                lines[idx] = f'version = {version!r}\n'
            elif match.group(1) == 'release':
                lines[idx] = f'release = {version!r}\n'
            elif match.group(1) == 'copyright':
                lines[idx] = f"copyright = '({date.year}, CNES/CLS)'\n"

    with open(conf, 'w', encoding='utf-8') as stream:
        stream.write(''.join(lines))

    # Finally, write the file containing the version number.
    with open(module, 'w', encoding='utf-8') as handler:
        handler.write(f'''# Copyright (c) {date.year} CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Get software version information
================================
"""


def release() -> str:
    """Returns the software version number"""
    return "{version}"


def date() -> str:
    """Returns the creation date of this release"""
    return "{date:%d %B %Y}"
''')
    return version


# pylint: disable=too-few-public-methods
class CMakeExtension(setuptools.Extension):
    """Python extension to build."""

    def __init__(self, name):
        super().__init__(name, sources=[])

    # pylint: enable=too-few-public-methods


# pylint: disable=too-many-instance-attributes
class BuildExt(setuptools.command.build_ext.build_ext):
    """Build everything needed to install."""
    user_options = setuptools.command.build_ext.build_ext.user_options
    user_options += [
        ('build-unittests', None, 'Build the unit tests of the C++ extension'),
        ('c-compiler=', None, 'Preferred C compiler'),
        ('cmake-args=', None, 'Additional arguments for CMake'),
        ('code-coverage', None, 'Enable coverage reporting'),
        ('cxx-compiler=', None, 'Preferred C++ compiler'),
        ('generator=', None, 'Selected CMake generator'),
        ('mkl=', None, 'Using MKL as BLAS library'),
        ('reconfigure', None, 'Forces CMake to reconfigure this project')
    ]

    boolean_options = setuptools.command.build_ext.build_ext.boolean_options
    boolean_options += ['mkl']

    def initialize_options(self) -> None:
        """Set default values for all the options that this command
        supports."""
        super().initialize_options()
        self.build_unittests = None
        self.code_coverage = None
        self.c_compiler = None
        self.cmake_args = None
        self.cxx_compiler = None
        self.generator = None
        self.mkl = None
        self.reconfigure = None

    def finalize_options(self) -> None:
        """Set final values for all the options that this command supports."""
        super().finalize_options()
        if self.code_coverage is not None and platform.system() == 'Windows':
            raise RuntimeError('Code coverage is not supported on Windows')

    def run(self) -> None:
        """Carry out the action."""
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    @staticmethod
    def set_conda_mklroot() -> None:
        """Set the default MKL path in Anaconda's environment."""
        mkl_header = pathlib.Path(sys.prefix, 'include', 'mkl.h')
        if not mkl_header.exists():
            mkl_header = pathlib.Path(sys.prefix, 'Library', 'include',
                                      'mkl.h')

        if mkl_header.exists():
            os.environ['MKLROOT'] = sys.prefix

    @staticmethod
    def conda_prefix() -> str | None:
        """Returns the conda prefix."""
        if 'CONDA_PREFIX' in os.environ:
            return os.environ['CONDA_PREFIX']
        return None

    def set_cmake_user_options(self) -> list[str]:
        """Sets the options defined by the user."""
        result = []

        conda_prefix = self.conda_prefix()

        if self.c_compiler is not None:
            result.append('-DCMAKE_C_COMPILER=' + self.c_compiler)

        if self.cxx_compiler is not None:
            result.append('-DCMAKE_CXX_COMPILER=' + self.cxx_compiler)

        if conda_prefix is not None:
            result.append('-DCMAKE_PREFIX_PATH=' + conda_prefix)

        if self.mkl:
            self.set_conda_mklroot()

        return result

    def get_config(self) -> str:
        """Returns the configuration to use."""
        cfg: str
        if self.debug:
            cfg = 'Debug'
        elif self.code_coverage:
            cfg = 'RelWithDebInfo'
        else:
            cfg = 'Release'
        return cfg

    def cmake_arguments(self, cfg: str, extdir: str) -> list[str]:
        """Returns the cmake arguments."""
        cmake_args: list[str] = [
            '-DCMAKE_BUILD_TYPE=' + cfg,
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPython3_EXECUTABLE=' + sys.executable,
            *self.set_cmake_user_options()
        ]

        if platform.python_implementation() == 'PyPy':
            cmake_args.append('-DPython3_FIND_IMPLEMENTATIONS=PyPy')
        elif 'Pyston' in sys.version:
            cmake_args.append('-DPython3_INCLUDE_DIR=' +
                              sysconfig.get_path('include'))
        return cmake_args

    def build_cmake(self, ext) -> None:
        """Execute cmake to build the Python extension."""
        # These dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(WORKING_DIRECTORY, self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = str(
            pathlib.Path(self.get_ext_fullpath(ext.name)).parent.resolve())
        cfg = self.get_config()
        cmake_args = self.cmake_arguments(cfg, extdir)
        build_args = ['--config', cfg]

        is_windows = platform.system() == 'Windows'

        if self.generator is not None:
            cmake_args.append('-G' + self.generator)
        elif is_windows:
            cmake_args.append(
                '-G' + os.environ.get('CMAKE_GEN', 'Visual Studio 16 2019'))

        if not is_windows:
            build_args += ['--', f'-j{os.cpu_count()}']
            if platform.system() == 'Darwin':
                cmake_args += [
                    f'-DCMAKE_OSX_DEPLOYMENT_TARGET={OSX_DEPLOYMENT_TARGET}'
                ]
            if self.code_coverage:
                cmake_args += ['-DCODE_COVERAGE=ON']
        else:
            cmake_args += [
                '-DCMAKE_GENERATOR_PLATFORM=x64',
                f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}',
            ]
            build_args += ['--', '/m']

        if self.cmake_args:
            cmake_args.extend(self.cmake_args.split())

        os.chdir(str(build_temp))

        # Has CMake ever been executed?
        if pathlib.Path(build_temp, 'CMakeFiles',
                        'TargetDirectories.txt').exists():
            # The user must force the reconfiguration
            configure = self.reconfigure is not None
        else:
            configure = True

        if configure:
            self.spawn(['cmake', str(WORKING_DIRECTORY), *cmake_args])
        if not self.dry_run:
            cmake_cmd = ['cmake', '--build', '.']
            if self.build_unittests is None:
                cmake_cmd += ['--target', 'core']
            self.spawn(cmake_cmd + build_args)
        os.chdir(str(WORKING_DIRECTORY))

    # pylint: enable=too-many-instance-attributes


class CxxTestRunner(setuptools.Command):
    """Compile and launch the C++ tests."""
    description: ClassVar[str] = 'run the C++ tests'
    user_options: ClassVar[list[tuple[str, str | None, str]]] = []

    def initialize_options(self):
        """Set default values for all the options that this command
        supports."""
        if platform.system() == 'Windows':
            raise RuntimeError('Code coverage is not supported on Windows')

    def finalize_options(self):
        """Set final values for all the options that this command supports."""

    def run(self):
        """Run tests."""
        # Directory used during the generating the C++ extension.
        tempdir = distutils_dirname('temp')

        # Navigate to the directory containing the C++ tests and run them.
        os.chdir(str(tempdir / 'src' / 'pyinterp' / 'core' / 'tests'))
        self.spawn(['ctest', '--output-on-failure'])

        # File containing the coverage report.
        coverage_lcov = str(
            pathlib.Path(tempdir.parent.parent, 'coverage_cpp.lcov'))

        # Collect coverage data from python/C++ unit tests
        self.spawn([
            'lcov', '--capture', '--directory',
            str(tempdir), '--output-file', coverage_lcov
        ])


class SDist(setuptools.command.sdist.sdist):
    """Custom sdist command that copies the pytest configuration file into the
    package."""
    user_options = setuptools.command.sdist.sdist.user_options

    def run(self):
        """Carry out the action."""
        source = WORKING_DIRECTORY.joinpath('conftest.py')
        target = WORKING_DIRECTORY.joinpath('src', 'pyinterp', 'conftest.py')
        source.rename(target)
        try:
            super().run()
        finally:
            target.rename(source)


def long_description():
    """Reads the README file."""
    with pathlib.Path(WORKING_DIRECTORY,
                      'README.rst').open(encoding='utf-8') as stream:
        return stream.read()


def typehints():
    """Get the list of type information files."""
    pyi = []
    for root, _, files in os.walk(WORKING_DIRECTORY):
        pyi += [
            str(pathlib.Path(root, item).relative_to(WORKING_DIRECTORY))
            for item in files if item.endswith('.pyi')
        ]
    return [(str(pathlib.Path('pyinterp', 'core')), pyi)]


def main():
    """Main function."""
    install_requires = ['dask', 'numpy', 'xarray >= 0.13']
    setuptools.setup(
        author='CNES/CLS',
        author_email='fbriol@gmail.com',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Topic :: Scientific/Engineering :: Physics',
            'Natural Language :: English',
            'Operating System :: POSIX',
            'Operating System :: MacOS',
            'Operating System :: Microsoft :: Windows',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: Python :: 3.13',
        ],
        cmdclass={
            'build_ext': BuildExt,
            'sdist': SDist,
            'gtest': CxxTestRunner,
        },
        data_files=typehints(),
        description='Interpolation of geo-referenced data for Python.',
        ext_modules=[CMakeExtension(name='pyinterp.core')],
        install_requires=install_requires,
        keywords='interpolation, geospatial, geohash, geodetic',
        license='BSD-3-Clause',
        license_files=('LICENSE', ),
        long_description=long_description(),
        long_description_content_type='text/x-rst',
        name='pyinterp',
        package_data={
            'pyinterp': ['py.typed', 'core/*.pyi', 'core/geohash/*.pyi'],
            'pyinterp.tests': ['dataset/*'],
        },
        package_dir={'': 'src'},
        packages=setuptools.find_namespace_packages(
            where='src',
            exclude=['pyinterp.core*'],
        ),
        platforms=['POSIX', 'MacOS', 'Windows'],
        python_requires='>=3.10',
        url='https://github.com/CNES/pangeo-pyinterp',
        version=revision(),
        zip_safe=False,
    )


if __name__ == '__main__':
    if platform.system() == 'Darwin':
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = OSX_DEPLOYMENT_TARGET
    main()
