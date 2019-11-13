# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""This script is the entry point for building, distributing and installing
this module using distutils/setuptools."""
import datetime
import distutils.command.build
import pathlib
import platform
import re
import subprocess
import os
import shlex
import sys
import sysconfig
import setuptools
import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.test
import pytest

# Check Python requirement
MAJOR = sys.version_info[0]
MINOR = sys.version_info[1]
if not (MAJOR >= 3 and MINOR >= 6):
    raise RuntimeError("Python %d.%d is not supported, "
                       "you need at least Python 3.6." % (MAJOR, MINOR))

# Working directory
WORKING_DIRECTORY = pathlib.Path(__file__).parent.absolute()


def build_dirname(extname=None):
    """Returns the name of the build directory"""
    extname = '' if extname is None else os.sep.join(extname.split(".")[:-1])
    return str(
        pathlib.Path(WORKING_DIRECTORY, "build",
                     "lib.%s-%d.%d" % (sysconfig.get_platform(), MAJOR, MINOR),
                     extname))


def execute(cmd):
    """Executes a command and returns the lines displayed on the standard
    output"""
    process = subprocess.Popen(cmd,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    return process.stdout.read().decode()


def update_meta(path, version):
    """Updating the version number description in conda/meta.yaml."""
    with open(path, "r") as stream:
        lines = stream.readlines()
    pattern = re.compile(r'{% set version = ".*" %}')

    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if match is not None:
            lines[idx] = '{%% set version = "%s" %%}\n' % version

    with open(path, "w") as stream:
        stream.write("".join(lines))


def update_environment(path, version):
    """Updating the version number desciption in conda environment"""
    with open(path, 'r') as stream:
        lines = stream.readlines()
    pattern = re.compile(r'(\s+-\s+pyinterp)\s*>=\s*(.+)')

    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if match is not None:
            lines[idx] = "%s>=%s\n" % (match.group(1), version)

    with open(path, "w") as stream:
        stream.write("".join(lines))


def revision():
    """Returns the software version"""
    os.chdir(WORKING_DIRECTORY)
    module = pathlib.Path(WORKING_DIRECTORY, 'src', 'pyinterp', 'version.py')
    stdout = execute("git describe --tags --dirty --long --always").strip()
    pattern = re.compile(r'([\w\d\.]+)-(\d+)-g([\w\d]+)(?:-(dirty))?')
    match = pattern.search(stdout)

    # If the information is unavailable (execution of this function outside the
    # development environment), file creation is not possible
    if not stdout:
        pattern = re.compile(r'return "(\d+\.\d+\.\d+)"')
        with open(module, "r") as stream:
            for line in stream:
                match = pattern.search(line)
                if match:
                    return match.group(1)
        raise AssertionError()

    # No tag already registred
    if match is None:
        pattern = re.compile(r'([\w\d]+)(?:-(dirty))?')
        match = pattern.search(stdout)
        version = "0.0"
        sha1 = match.group(1)
    else:
        version = match.group(1)
        sha1 = match.group(3)

    stdout = execute("git log  %s -1 --format=\"%%H %%at\"" % sha1)
    stdout = stdout.strip().split()
    date = datetime.datetime.utcfromtimestamp(int(stdout[1]))

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
    with open(conf, "r") as stream:
        lines = stream.readlines()
    pattern = re.compile(r'(\w+)\s+=\s+(.*)')

    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if match is not None:
            if match.group(1) == 'version':
                lines[idx] = "version = %r\n" % version
            elif match.group(1) == 'release':
                lines[idx] = "release = %r\n" % version
            elif match.group(1) == 'copyright':
                lines[idx] = "copyright = '(%s, CNES/CLS)'\n" % date.year

    with open(conf, "w") as stream:
        stream.write("".join(lines))

    # Finally, write the file containing the version number.
    with open(module, 'w') as handler:
        handler.write('''"""
Get software version information
================================
"""


def release() -> str:
    """Returns the software version number"""
    return "{version}"


def date() -> str:
    """Returns the creation date of this release"""
    return "{date}"
'''.format(version=version, date=date.strftime("%d %B %Y")))
    return version


# pylint: disable=too-few-public-methods
class CMakeExtension(setuptools.Extension):
    """Python extension to build"""
    def __init__(self, name):
        super(CMakeExtension, self).__init__(name, sources=[])

    # pylint: enable=too-few-public-methods


class BuildExt(setuptools.command.build_ext.build_ext):
    """Build the Python extension using cmake"""

    #: Preferred BOOST root
    BOOST_ROOT = None

    #: Build the unit tests of the C++ extension
    BUILD_INITTESTS = None

    #: Enable coverage reporting
    CODE_COVERAGE = None

    #: Preferred C++ compiler
    CXX_COMPILER = None

    #: Preferred Eigen root
    EIGEN3_INCLUDE_DIR = None

    #: Preferred GSL root
    GSL_ROOT = None

    #: Run CMake to configure this project
    RECONFIGURE = None

    def run(self):
        """A command's raison d'etre: carry out the action"""
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    @staticmethod
    def gsl():
        """Get the default boost path in Anaconda's environnement."""
        gsl_root = sys.prefix
        if os.path.exists(os.path.join(gsl_root, "include", "gsl")):
            return "-DGSL_ROOT_DIR=" + gsl_root
        gsl_root = os.path.join(sys.prefix, "Library")
        if not os.path.exists(os.path.join(gsl_root, "include", "gsl")):
            raise RuntimeError(
                "Unable to find the GSL library in the conda distribution "
                "used.")
        return "-DGSL_ROOT_DIR=" + gsl_root

    @staticmethod
    def boost():
        """Get the default boost path in Anaconda's environnement."""
        # Do not search system for Boost & disable the search for boost-cmake
        boost_option = "-DBoost_NO_SYSTEM_PATHS=TRUE " \
            "-DBoost_NO_BOOST_CMAKE=TRUE"
        boost_root = sys.prefix
        if os.path.exists(os.path.join(boost_root, "include", "boost")):
            return "{boost_option} -DBOOST_ROOT={boost_root}".format(
                boost_root=boost_root, boost_option=boost_option).split()
        boost_root = os.path.join(sys.prefix, "Library", "include")
        if not os.path.exists(boost_root):
            raise RuntimeError(
                "Unable to find the Boost library in the conda distribution "
                "used.")
        return "{boost_option} -DBoost_INCLUDE_DIR={boost_root}".format(
            boost_root=boost_root, boost_option=boost_option).split()

    @staticmethod
    def eigen():
        """Get the default Eigen3 path in Anaconda's environnement."""
        eigen_include_dir = os.path.join(sys.prefix, "include", "eigen3")
        if os.path.exists(eigen_include_dir):
            return "-DEIGEN3_INCLUDE_DIR=" + eigen_include_dir
        eigen_include_dir = os.path.join(sys.prefix, "Library", "include",
                                         "eigen3")
        if not os.path.exists(eigen_include_dir):
            eigen_include_dir = os.path.dirname(eigen_include_dir)
        if not os.path.exists(eigen_include_dir):
            raise RuntimeError(
                "Unable to find the Eigen3 library in the conda distribution "
                "used.")
        return "-DEIGEN3_INCLUDE_DIR=" + eigen_include_dir

    @staticmethod
    def is_conda():
        """Detect if the Python interpreter is part of a conda distribution."""
        result = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
        if not result:
            try:
                # pylint: disable=unused-import
                import conda
                # pylint: enable=unused-import
            except ImportError:
                result = False
            else:
                result = True
        return result

    def set_cmake_user_options(self):
        """Sets the options defined by the user."""
        is_conda = self.is_conda()
        result = []

        if self.CXX_COMPILER is not None:
            result.append("-DCMAKE_CXX_COMPILER=" + self.CXX_COMPILER)

        if self.BOOST_ROOT is not None:
            result.append("-DBOOSTROOT=" + self.BOOST_ROOT)
        elif is_conda:
            result += self.boost()

        if self.GSL_ROOT is not None:
            result.append("-DGSL_ROOT_DIR=" + self.GSL_ROOT)
        elif is_conda:
            result.append(self.gsl())

        if self.EIGEN3_INCLUDE_DIR is not None:
            result.append("-DEIGEN3_INCLUDE_DIR=" + self.EIGEN3_INCLUDE_DIR)
        elif is_conda:
            result.append(self.eigen())

        return result

    def build_cmake(self, ext):
        """Execute cmake to build the Python extension"""
        # These dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(WORKING_DIRECTORY, self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = build_dirname(ext.name)

        cfg = 'Debug' if self.debug or self.CODE_COVERAGE else 'Release'

        cmake_args = [
            "-DCMAKE_BUILD_TYPE=" + cfg, "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" +
            str(extdir), "-DPYTHON_EXECUTABLE=" + sys.executable
        ] + self.set_cmake_user_options()

        build_args = ['--config', cfg]

        if platform.system() != 'Windows':
            build_args += ['--', '-j%d' % os.cpu_count()]
            if platform.system() == 'Darwin':
                cmake_args += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=10.14']
            if self.CODE_COVERAGE:
                cmake_args += ["-DCODE_COVERAGE=ON"]
        else:
            cmake_args += [
                '-G', 'Visual Studio 15 2017',
                '-DCMAKE_GENERATOR_PLATFORM=x64',
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), extdir)
            ]
            build_args += ['--', '/m']
            if self.verbose:
                build_args += ['/verbosity:n']

        if self.verbose:
            build_args.insert(0, "--verbose")

        os.chdir(str(build_temp))

        # Has CMake ever been executed?
        if pathlib.Path(build_temp, "CMakeFiles",
                        "TargetDirectories.txt").exists():
            # The user must force the reconfiguration
            configure = self.RECONFIGURE is not None
        else:
            configure = True

        if configure:
            self.spawn(['cmake', str(WORKING_DIRECTORY)] + cmake_args)
        if not self.dry_run:
            cmake_cmd = ['cmake', '--build', '.']
            if self.BUILD_INITTESTS is None:
                cmake_cmd += ['--target', 'core']
            self.spawn(cmake_cmd + build_args)
        os.chdir(str(WORKING_DIRECTORY))


class Build(distutils.command.build.build):
    """Build everything needed to install"""
    user_options = distutils.command.build.build.user_options
    user_options += [
        ('boost-root=', None, 'Preferred Boost installation prefix'),
        ('build-unittests', None, "Build the unit tests of the C++ extension"),
        ('reconfigure', None, 'Forces CMake to reconfigure this project'),
        ('code-coverage', None, 'Enable coverage reporting'),
        ('cxx-compiler=', None, 'Preferred C++ compiler'),
        ('eigen-root=', None, 'Preferred Eigen3 include directory'),
        ('gsl-root=', None, 'Preferred GSL installation prefix')
    ]

    def initialize_options(self):
        """Set default values for all the options that this command supports"""
        super().initialize_options()
        self.boost_root = None
        self.build_unittests = None
        self.code_coverage = None
        self.cxx_compiler = None
        self.eigen_root = None
        self.gsl_root = None
        self.reconfigure = None

    def finalize_options(self):
        """Set final values for all the options that this command supports"""
        super().finalize_options()
        if self.code_coverage is not None and platform.system() == 'Windows':
            raise RuntimeError("Code coverage is not supported on Windows")

    def run(self):
        """A command's raison d'etre: carry out the action"""
        if self.boost_root is not None:
            BuildExt.BOOST_ROOT = self.boost_root
        if self.build_unittests is not None:
            BuildExt.BUILD_INITTESTS = self.build_unittests
        if self.cxx_compiler is not None:
            BuildExt.CXX_COMPILER = self.cxx_compiler
        if self.code_coverage is not None:
            if platform.system() == 'Windows':
                raise RuntimeError("Code coverage is not supported on Windows")
            BuildExt.CODE_COVERAGE = self.code_coverage
        if self.eigen_root is not None:
            BuildExt.EIGEN3_INCLUDE_DIR = self.eigen_root
        if self.gsl_root is not None:
            BuildExt.GSL_ROOT = self.gsl_root
        if self.reconfigure is not None:
            BuildExt.RECONFIGURE = True
        super().run()


class Test(setuptools.command.test.test):
    """Test runner"""
    user_options = [('ext-coverage', None,
                     "Generate C++ extension coverage reports"),
                    ("pytest-args=", None, "Arguments to pass to pytest")]

    def initialize_options(self):
        """Set default values for all the options that this command
        supports"""
        super().initialize_options()
        self.ext_coverage = None
        self.pytest_args = None

    def finalize_options(self):
        """Set final values for all the options that this command supports"""
        dirname = pathlib.Path(pathlib.Path(__file__).absolute().parent)
        rootdir = "--rootdir=" + str(dirname)
        if self.pytest_args is None:
            self.pytest_args = ''
        self.pytest_args = rootdir + " tests " + self.pytest_args

    @staticmethod
    def tempdir():
        """Gets the build directory of the extension"""
        return pathlib.Path(
            WORKING_DIRECTORY, "build",
            "temp.%s-%d.%d" % (sysconfig.get_platform(), MAJOR, MINOR))

    def run_tests(self):
        """Run tests"""
        sys.path.insert(0, build_dirname())

        errno = pytest.main(
            shlex.split(self.pytest_args,
                        posix=platform.system() != 'Windows'))
        if errno:
            sys.exit(errno)

        # Directory used during the generating the C++ extension.
        tempdir = self.tempdir()

        # We work in the extension generation directory (CMake directory)
        os.chdir(str(tempdir))

        # If the C++ unit tests have been generated, they are executed.
        if pathlib.Path(tempdir, "src", "pyinterp", "core", "tests",
                        "test_axis").exists():
            self.spawn(["ctest", "--output-on-failure"])

        # Generation of the code coverage of the C++ extension?
        if not self.ext_coverage:
            return

        # Directory for writing the HTML coverage report.
        htmllcov = str(pathlib.Path(tempdir.parent.parent, "htmllcov"))

        # File containing the coverage report.
        coverage_info = str(pathlib.Path(tempdir, "coverage.info"))

        # Collect coverage data from python/C++ unit tests
        self.spawn([
            "lcov", "--capture", "--directory",
            str(tempdir), "--output-file", coverage_info
        ])

        # The coverage of third-party libraries is removed.
        self.spawn([
            'lcov', '-r', coverage_info, "*/Xcode.app/*", "*/third_party/*",
            "*/boost/*", "*/eigen3/*", "*/tests/*", "*/usr/*", '--output-file',
            coverage_info
        ])

        # Finally, we generate the HTML coverage report.
        self.spawn(["genhtml", coverage_info, "--output-directory", htmllcov])


def long_description():
    """Reads the README file"""
    with open(pathlib.Path(WORKING_DIRECTORY, "README.md")) as stream:
        return stream.read()


def main():
    """Main function"""
    setuptools.setup(
        author='CNES/CLS',
        author_email='fbriol@gmail.com',
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Topic :: Scientific/Engineering :: Physics",
            "License :: OSI Approved :: BSD License",
            "Natural Language :: English", "Operating System :: POSIX",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8"
        ],
        cmdclass={
            'build': Build,
            'build_ext': BuildExt,
            'test': Test
        },
        description='Interpolation of geo-referenced data for Python.',
        ext_modules=[CMakeExtension(name="pyinterp.core")],
        install_requires=["numpy", "xarray"],
        license="BSD License",
        long_description=long_description(),
        long_description_content_type='text/markdown',
        name='pyinterp',
        package_dir={'': 'src'},
        packages=setuptools.find_namespace_packages(where='src',
                                                    exclude=['*core*']),
        platforms=['POSIX', 'MacOS', 'Windows'],
        python_requires='>=3.6',
        tests_require=["netCDF4", "numpy", "pytest", "xarray>=0.13"],
        url='https://github.com/CNES/pangeo-pyinterp',
        version=revision(),
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
