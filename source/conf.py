# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html



import os
import sys

import sphinx_rtd_theme


sys.path.insert(0,os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DREAMS_MC'
copyright = '2024, Rabindra Khadka'
author = 'Rabindra Khadka'
release = '08/05/2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = 'English'

# If true, keep warnings as "system message" paragraphs in the built documents.
keep_warnings = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.imgmath",
    "sphinx.ext.napoleon",
]

# Add any paths that contain custom static files (such as style sheets) here,
html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------


# Output file base name for HTML help builder.
htmlhelp_basename = "DREAMSdoc"



# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright


# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}