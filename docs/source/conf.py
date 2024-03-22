# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import shutil
import sys

# sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.join(os.path.abspath("../.."), "gpm"))
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# -- Project information -----------------------------------------------------

project = "gpm_api"
copyright = "Gionata Ghiggi"
author = "Gionata Ghiggi"


# -- Copy Jupyter Notebook Tutorials------------------------------------------
root_path = os.path.dirname(os.path.dirname(os.getcwd()))
filenames = [
    "tutorial_02_IMERG.ipynb",
    "tutorial_02_PMW_1C.ipynb",
    "tutorial_02_PMW_2A.ipynb",
    "tutorial_02_RADAR_2A.ipynb",
]
for filename in filenames:
    in_path = os.path.join(root_path, "tutorials", filename)
    out_path = os.path.join(os.getcwd(), "tutorials", filename)
    shutil.copyfile(in_path, out_path)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    # "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "sphinx_mdinclude",
    "sphinxcontrib.youtube",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_title = "GPM-API"
html_theme_options = {
    "repository_url": "https://github.com/ghiggi/gpm_api",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_edit_page_button": True,
    # "use_source_button": True,
    "use_issues_button": True,
    # "use_repository_button": True,
    "use_download_button": True,
    # "use_sidenotes": True,
    "show_toc_level": 2,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]


# -- Automatically run apidoc to generate rst from code ----------------------
# https://github.com/readthedocs/readthedocs.org/issues/1139
def run_apidoc(_):
    from sphinx.ext.apidoc import main

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    cur_dir = os.path.abspath(os.path.dirname(__file__))

    module_dir = os.path.join(cur_dir, "..", "..", "gpm")
    output_dir = os.path.join(cur_dir, "api")
    exclude = [os.path.join(module_dir, "tests")]
    main(["-f", "-o", output_dir, module_dir, *exclude])


def setup(app):
    app.connect("builder-inited", run_apidoc)
