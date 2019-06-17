import distutils.command.build
import pathlib
import platform
import setuptools
import setuptools.command.build_ext
import setuptools.command.install
import os
import sys
import sysconfig


class CMakeExtension(setuptools.Extension):
    """Python extension to build"""

    def __init__(self, name):
        super(CMakeExtension, self).__init__(name, sources=[])


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
        eigen_include_dir = os.path.join(sys.prefix, "Library", "include")
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
            return 'conda-bld' in sysconfig.get_config_var("abs_srcdir")
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
        ]

        if self.CXX_COMPILER is not None:
            cmake_args.append("-DCMAKE_CXX_COMPILER=" + self.CXX_COMPILER)

        is_conda = self.is_conda()

        if self.BOOST_ROOT is not None:
            cmake_args.append("-DBOOST_ROOT=" + self.BOOST_ROOT)
        elif is_conda:
            cmake_args.append(self.boost())

        if self.GSL_ROOT is not None:
            cmake_args.append("-DGSL_ROOT_DIR=" + self.GSL_ROOT)
        elif is_conda:
            cmake_args.append(self.gsl())

        if self.EIGEN3_INCLUDE_DIR is not None:
            cmake_args.append("-DEIGEN3_INCLUDE_DIR=" +
                              self.EIGEN3_INCLUDE_DIR)
        elif is_conda:
            cmake_args.append(self.eigen())

        build_args = ['--config', cfg]

        if platform.system() != 'Windows':
            build_args += ['--', '-j%d' % os.cpu_count()]
            cmake_args += [
                '-DCMAKE_BUILD_TYPE=' + cfg
            ]
            if platform.system() == 'Darwin':
                cmake_args += [
                    '-DCMAKE_OSX_DEPLOYMENT_TARGET=10.14'
                ]
        else:
            cmake_args += [
                '-DCMAKE_GENERATOR_PLATFORM=x64',
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), extdir)
            ]
            build_args += ['--', '/m']

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        #+ ['/verbosity:n']
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


def main():
    setuptools.setup(name='pyinterp',
                     version='0.1',
                     description='TODO',
                     url='TODO',
                     author='CLS',
                     license="Proprietary",
                     ext_modules=[CMakeExtension(name="pyinterp.core")],
                     packages=setuptools.find_namespace_packages(
                         include=['pyinterp*'], exclude=['*core*']),
                     cmdclass={
                         'build': Build,
                         'build_ext': BuildExt
                     },
                     zip_safe=False)


if __name__ == "__main__":
    main()