#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------
import pathlib
import sysconfig
import sys

HERE = pathlib.Path(__file__).absolute().parent
ROOT_DIRECTORY = HERE.parent.parent
MAJOR = sys.version_info[0]
MINOR = sys.version_info[1]


def get_build_dirname():
    """Returns the name of the build directory."""
    path = pathlib.Path(
        ROOT_DIRECTORY, 'build',
        'lib.%s-%d.%d' % (sysconfig.get_platform(), MAJOR, MINOR))
    if path.exists():
        return path
    return pathlib.Path(
        ROOT_DIRECTORY, 'build',
        f'lib.{sysconfig.get_platform()}-{sys.implementation.cache_tag}')


build_dirname = get_build_dirname()
if build_dirname.exists():
    sys.path.insert(0, str(build_dirname))

# -- Project information -----------------------------------------------------

project = 'pyinterp'
copyright = '(2025, CNES/CLS)'
author = 'CNES/CLS'

# The short X.Y version
version = '2025.3.0'
# The full version, including alpha/beta/rc tags
release = '2025.3.0'

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx.ext.autosummary',
    'sphinx.ext.mathjax', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery', 'sphinx_inline_tabs'
]

autosummary_generate = True

numpydoc_class_members_toctree = True
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_title = 'PyInterp'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_theme_options = {
    'light_logo': 'pyinterp-light.svg',
    'dark_logo': 'pyinterp-dark.svg',
    'sidebar_hide_name': True,
}

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'pyinterpdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'pyinterp.tex', 'PyInterp Documentation', 'CLS', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'pyinterp', 'PyInterp Documentation', [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'pyinterp', 'PyInterp Documentation', author, 'pyinterp',
     'One line description of project.', 'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# -- Extension configuration -------------------------------------------------
sphinx_gallery_conf = {
    'examples_dirs': [HERE.parent.parent.joinpath('examples')],
    'filename_pattern': r'[\\\/]ex_',
    'pypandoc': False,
    'binder': {
        'org': 'CNES',
        'repo': 'pangeo-pyinterp',
        'branch': 'master',
        'binderhub_url': 'https://mybinder.org',
        'dependencies':
        [HERE.joinpath('..', '..', 'binder', 'environment.yml')],
        'use_jupyter_lab': True,
    }
}

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'dask': ('https://docs.dask.org/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
}

README = HERE.joinpath('readme.rst')

if not README.exists():
    with HERE.joinpath('..', '..', 'README.rst').open() as stream:
        contents = stream.read()

    with README.open('w') as stream:
        stream.write(contents)
