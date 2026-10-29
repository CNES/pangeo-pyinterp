"""Configuration file for the Sphinx documentation builder."""

import datetime
import importlib.util
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

# Make _gallery_order importable: sphinx-gallery resolves the sort keys by name.
sys.path.insert(0, str(HERE))

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
html_title = "pyinterp"
html_static_path = ["_static"]
# Files copied as they are to the root of the site, such as llms.txt.
html_extra_path = ["_extra"]
html_css_files = ["pyinterp.css"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "logo": {
        "image_light": "_static/pyinterp-light.svg",
        "image_dark": "_static/pyinterp-dark.svg",
        "alt_text": "pyinterp",
    },
    "repository_url": "https://github.com/CNES/pangeo-pyinterp",
    "use_repository_button": True,
    "max_navbar_depth": 2,  # Maximum depth of navigation tree
    # pydata-sphinx-theme adds a search field to its header unless this is set
    # explicitly; sphinx-book-theme already shows one in the sidebar.
    "navbar_persistent": [],
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
    # Run the examples with as many processes as sphinx-build -j, which
    # requires joblib.
    "parallel": importlib.util.find_spec("joblib") is not None,
    "pypandoc": False,
    "subsection_order": "_gallery_order.section_key",
    "default_thumb_file": str(HERE.joinpath("_static", "gallery-thumb.png")),
    "within_subsection_order": "_gallery_order.ExampleKey",
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

# Copy the README file into the docs/source directory, unless the copy is
# already current: a copy written once would go stale as the README changes.
_contents = HERE.joinpath("..", "..", "README.rst").read_text(encoding="utf-8")
if not README.exists() or README.read_text(encoding="utf-8") != _contents:
    README.write_text(_contents, encoding="utf-8")


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
