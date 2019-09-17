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
import sys
import setuptools
import setuptools.command.build_ext
import setuptools.command.install

# Check Python requirement
MAJOR = sys.version_info[0]
MINOR = sys.version_info[1]
if not (MAJOR >= 3 and MINOR >= 6):
    raise RuntimeError("Python %d.%d is not supported, "
                       "you need at least Python 3.6." % (MAJOR, MINOR))


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


def revision():
    """Returns the software version"""
    cwd = pathlib.Path().absolute()
    module = os.path.join(cwd, 'src', 'pyinterp', 'version.py')
    stdout = execute("git describe --tags --dirty --long --always").strip()
    pattern = re.compile(r'([\w\d\.]+)-(\d+)-g([\w\d]+)(?:-(dirty))?')
    match = pattern.search(stdout)

    # If the information is unavailable (execution of this function outside the
    # development environment), file creation is not possible
    if not stdout:
        pattern = re.compile(r'\s+result = "(.*)"')
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

    # This file is not present in the distribution, but only in the GIT
    # repository of the source code.
    meta = os.path.join(cwd, 'conda', 'meta.yaml')
    if os.path.exists(meta):
        update_meta(meta, version)

    # Updating the version number description for sphinx
    conf = os.path.join(cwd, 'docs', 'source', 'conf.py')
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


def release(full: bool = False) -> str:
    """Returns the software version number"""
    result = "{version}"
    if full:
        result += " ({date})"
    return result
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

    #: Preferred C++ compiler
    CXX_COMPILER = None

    #: Preferred BOOST root
    BOOST_ROOT = None

    #: Preferred GSL root
    GSL_ROOT = None

    #: Preferred Eigen root
    EIGEN3_INCLUDE_DIR = None

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
        boost_root = sys.prefix
        if os.path.exists(os.path.join(boost_root, "include", "boost")):
            return "-DBOOST_ROOT=" + boost_root
        boost_root = os.path.join(sys.prefix, "Library", "include")
        if not os.path.exists(boost_root):
            raise RuntimeError(
                "Unable to find the Boost library in the conda distribution "
                "used.")
        return "-DBoost_INCLUDE_DIR=" + boost_root

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
            result.append("-DBOOST_ROOT=" + self.BOOST_ROOT)
        elif is_conda:
            result.append(self.boost())

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
        cwd = pathlib.Path().absolute()

        # These dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(
            ext.name)).absolute().parent

        cfg = 'Debug' if self.debug else 'Release'

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(extdir),
            "-DPYTHON_EXECUTABLE=" + sys.executable
        ] + self.set_cmake_user_options()

        build_args = ['--config', cfg]

        if platform.system() != 'Windows':
            build_args += ['--', '-j%d' % os.cpu_count()]
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            if platform.system() == 'Darwin':
                cmake_args += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=10.14']
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
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.', '--target', 'core'] +
                       build_args)
        os.chdir(str(cwd))


class Build(distutils.command.build.build):
    """Build everything needed to install"""
    user_options = distutils.command.build.build.user_options
    user_options += [
        ('boost-root=', 'u', 'Preferred Boost installation prefix'),
        ('gsl-root=', 'g', 'Preferred GSL installation prefix'),
        ('eigen-root=', 'e', 'Preferred Eigen3 include directory'),
        ('cxx-compiler=', 'x', 'Preferred C++ compiler')
    ]

    def initialize_options(self):
        """Set default values for all the options that this command supports"""
        super().initialize_options()
        self.boost_root = None
        self.cxx_compiler = None
        self.eigen_root = None
        self.gsl_root = None

    def run(self):
        """A command's raison d'etre: carry out the action"""
        if self.boost_root is not None:
            BuildExt.BOOST_ROOT = self.boost_root
        if self.cxx_compiler is not None:
            BuildExt.CXX_COMPILER = self.cxx_compiler
        if self.gsl_root is not None:
            BuildExt.GSL_ROOT = self.gsl_root
        if self.eigen_root is not None:
            BuildExt.EIGEN3_INCLUDE_DIR = self.eigen_root
        super().run()


def long_description():
    """Reads the README file"""
    with open("README.md") as stream:
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
            "Programming Language :: Python :: 3.7"
        ],
        cmdclass={
            'build': Build,
            'build_ext': BuildExt
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
        tests_require=["netCDF4", "numpy", "xarray"],
        url='https://github.com/CNES/pangeo-pyinterp',
        version=revision(),
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
