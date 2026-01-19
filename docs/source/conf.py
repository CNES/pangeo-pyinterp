"""Configuration file for the Sphinx documentation builder."""

import datetime
import inspect
import pathlib
import sys
import sysconfig
from types import CoroutineType, FunctionType, MethodType
from typing import Any, TypeIs
from collections.abc import Callable

from sphinx.util import inspect as sphinx_inspect


# Where this file is located
HERE = pathlib.Path(__file__).absolute().parent

# The root directory of the project
ROOT_DIRECTORY = HERE.parent.parent

# Python major and minor version
MAJOR = sys.version_info[0]
MINOR = sys.version_info[1]

# The README file located here is a copy of the file located in the root
# directory of the project.
README = HERE.joinpath("readme.rst")


def get_build_dirname() -> pathlib.Path:
    """Return the name of the build directory."""
    path = pathlib.Path(
        ROOT_DIRECTORY,
        "build",
        "lib.%s-%d.%d" % (sysconfig.get_platform(), MAJOR, MINOR),
    )
    if path.exists():
        return path
    return pathlib.Path(
        ROOT_DIRECTORY,
        "build",
        f"lib.{sysconfig.get_platform()}-{sys.implementation.cache_tag}",
    )


class SubsectionSorter:
    """Sorts example files within a subsection based on a predefined order."""

    def __init__(self, *args: Any) -> None:  # noqa: ANN401
        """Initialize the sorter."""

    def __call__(self, fname: str) -> int:
        """Return the order index for a given filename."""
        order = {
            "ex_objects.py": 0,
            "ex_axis.py": 1,
            "ex_geodetic.py": 2,
            "ex_geohash.py": 3,
            "ex_1d.py": 4,
            "ex_2d.py": 5,
            "ex_3d.py": 6,
            "ex_4d.py": 7,
            "ex_unstructured.py": 8,
            "ex_orbit.py": 9,
            "pangeo_unstructured_grid.py": 10,
            "pangeo_time_series.py": 11,
            "ex_descriptive_statistics.py": 12,
            "ex_binning.py": 13,
            "ex_dateutils.py": 14,
            "ex_fill_undef.py": 15,
        }
        return order[pathlib.Path(fname).name]


def section_sorter(dirname: str) -> int:
    """Sorts sections based on a predefined order."""
    order = {
        "core": 0,
        "grid": 1,
        "stats": 2,
        "geo": 3,
        "utilities": 4,
        "orbit": 5,
        "pangeo": 6,
    }
    return order[pathlib.Path(dirname).name]


def _is_nanobind_func(obj: object) -> bool:
    """Check if object is a nanobind function or method."""
    type_name = type(obj).__name__
    return type_name in ("nb_func", "nb_method")


def _is_nanobind_method(obj: object) -> bool:
    """Check if object is a nanobind method."""
    return type(obj).__name__ == "nb_method"


# Store original isfunction
_SPHINX_INSPECT_IS_FUNCTION = sphinx_inspect.isfunction


def mpatch_isfunction(obj: object) -> TypeIs[FunctionType]:
    """Patch isfunction to recognize nb_func."""
    if _is_nanobind_func(obj):
        return True
    return _SPHINX_INSPECT_IS_FUNCTION(obj)


# Patch ismethod to recognize nb_method
def mpatch_ismethod(object: object) -> TypeIs[MethodType]:
    """Patch ismethod to recognize nb_method."""
    if _is_nanobind_method(object):
        return True
    return inspect.ismethod(object)


# Store original isclassmethod
_SPHINX_INSPECT_IS_CLASSMETHOD = sphinx_inspect.isclassmethod


def mpatch_isclassmethod(
    obj: object, cls: object = None, name: str | None = None
) -> TypeIs[classmethod[Any, Any, Any]]:
    """Patch isclassmethod to handle nanobind methods."""
    if _is_nanobind_method(obj):
        return False
    return _SPHINX_INSPECT_IS_CLASSMETHOD(obj, cls, name)


# Store original isstaticmethod
_SPHINX_INSPECT_IS_STATICMETHOD = sphinx_inspect.isstaticmethod


def mpatch_isstaticmethod(
    obj: object, cls: object = None, name: str | None = None
) -> TypeIs[staticmethod[Any, Any]]:
    """Patch isstaticmethod to handle nanobind functions."""
    # Check if it's a nanobind function in a class context
    if type(obj).__name__ == "nb_func" and cls is not None:
        return True
    return _SPHINX_INSPECT_IS_STATICMETHOD(obj, cls, name)


# Patch iscoroutinefunction
_SPHINX_INSPECT_IS_COROUTINEFUNCTION = getattr(
    sphinx_inspect, "iscoroutinefunction", None
)
if _SPHINX_INSPECT_IS_COROUTINEFUNCTION is not None:

    def mpatch_iscoroutinefunction(
        obj: Any,  # noqa: ANN401
    ) -> TypeIs[Callable[..., CoroutineType[Any, Any, Any]]]:
        """Patch iscoroutinefunction to handle nanobind functions."""
        if _is_nanobind_func(obj):
            return False
        assert _SPHINX_INSPECT_IS_COROUTINEFUNCTION is not None
        return _SPHINX_INSPECT_IS_COROUTINEFUNCTION(obj)


# Add the build directory to sys.path to be able to document the compiled
# extension modules (useful when building the docs in the development
# environment)
build_dirname = get_build_dirname()
if build_dirname.exists():
    sys.path.insert(0, str(build_dirname))

try:
    import pyinterp  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "pyinterp module could not be imported. "
        "Make sure the documentation is built "
        "after building the project."
    ) from exc

# Project information
project = "pyinterp"
copyright = f"({datetime.datetime.now().year}, CNES/CLS)"
author = "CNES/CLS"
release = pyinterp.__version__
version = ".".join(release.split(".")[:2])

# Sphinx extensions
extensions = [
    "sphinx_gallery.gen_gallery",
    "sphinx_inline_tabs",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
# Autosummary configuration
autosummary_generate = True

# Autodoc configuration for nanobind modules
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Tell autodoc to use type hints from stub files (.pyi)
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Don't fail on missing type hints
autodoc_type_aliases: dict[str, str] = {}

# Concatenate class and __init__ docstrings
autoclass_content = "both"

numpydoc_class_members_toctree = True
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"
html_title = "PyInterp"
html_static_path = ["_static"]
html_theme_options = {
    "logo": {
        "image_light": "_static/pyinterp-light.svg",
        "image_dark": "_static/pyinterp-dark.svg",
    },
    "repository_url": "https://github.com/CNES/pangeo-pyinterp",
    "use_repository_button": True,
    "max_navbar_depth": 2,  # Maximum depth of navigation tree
}

# Output file base name for HTML help builder.
htmlhelp_basename = "pyinterpdoc"

latex_elements: dict[str, str] = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "pyinterp.tex", "PyInterp Documentation", "CLS", "manual"),
]

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pyinterp", "PyInterp Documentation", [author], 1)]

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "pyinterp",
        "PyInterp Documentation",
        author,
        "pyinterp",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# Bibliographic Dublin Core info.
epub_title = project

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# Sphinx-Gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": [HERE.parent.parent.joinpath("examples")],
    "filename_pattern": r"[\\\/]ex_",
    "pypandoc": False,
    "subsection_order": section_sorter,
    "within_subsection_order": SubsectionSorter,
    "binder": {
        "org": "CNES",
        "repo": "pangeo-pyinterp",
        "branch": "master",
        "binderhub_url": "https://mybinder.org",
        "dependencies": [
            HERE.joinpath("..", "..", "binder", "environment.yml")
        ],
        "use_jupyter_lab": True,
    },
}

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# Ensure the README file is present in the docs/source directory
if not README.exists():
    with HERE.joinpath("..", "..", "README.rst").open() as stream:
        contents = stream.read()

    with README.open("w") as stream:
        stream.write(contents)


def setup(app: object) -> None:
    """Set up the Sphinx application."""
    # Sphinx inspects all objects in the module and tries to resolve their type
    # (attribute, function, class, module, etc.) by using its own functions in
    # `sphinx.util.inspect`. These functions misidentify certain nanobind
    # objects. We monkey patch those functions here.

    sphinx_inspect.isfunction = mpatch_isfunction
    sphinx_inspect.ismethod = mpatch_ismethod
    sphinx_inspect.isclassmethod = mpatch_isclassmethod
    sphinx_inspect.isstaticmethod = mpatch_isstaticmethod
    if _SPHINX_INSPECT_IS_COROUTINEFUNCTION is not None:
        sphinx_inspect.iscoroutinefunction = mpatch_iscoroutinefunction
