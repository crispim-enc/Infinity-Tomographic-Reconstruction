# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
# sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('../../src/'))
sys.path.insert(0, os.path.abspath('../../src/'))

project = 'Infinity-Tomographic-Reconstruction'
copyright = '2025, Pedro Encarnação'
author = 'Pedro Encarnação'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
# exclude_patterns = []
extensions = ["sphinx.ext.autodoc", 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'sphinx.ext.autosummary',
'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]
language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"
# html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_show_sourcelink = False
# Example of intersphinx mapping for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

html_logo = '_static/logo.png'

# Additional CSS for further customization
html_css_files = [
    'custom.css',  # We'll create this file next
]