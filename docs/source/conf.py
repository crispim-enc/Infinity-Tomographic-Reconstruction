# *******************************************************
# * FILE: conf.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
# sys.path.insert(0, os.path.abspath('../toor/'))
# sys.path.insert(0, os.path.abspath('../../toor/'))
sys.path.insert(0, os.path.abspath('../../src/toor/'))
from sphinx_gallery.sorting import FileNameSortKey
import matplotlib
matplotlib.use('agg')  # for headless image generation
def setup(app):
    app.add_css_file("custom.css")

project = 'Infinity-Tomographic-Reconstruction'
copyright = '2025, Pedro Encarnação'
license = 'CC BY-NC-SA 4.0'
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
'sphinx_gallery.gen_gallery',
'matplotlib.sphinxext.mathmpl',
          'matplotlib.sphinxext.plot_directive',
          'sphinx.ext.doctest',
"sphinx_design"
]
# 'examples_dirs': ['examples/EasyPETCT', 'examples/SiemensIntevoBoldSPECTCT'],   # path to your example scripts
#     'gallery_dirs': ['auto_examples/EasyPETCT', 'auto_examples/SiemensIntevoBoldSPECTCT'],
sphinx_gallery_conf = {
    'examples_dirs': 'examples',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # where to save gallery generated pages
    'filename_pattern': r'.*\.py$',
'within_subsection_order': FileNameSortKey,
   'backreferences_dir' :None,
}
    # {
    #     'examples_dirs': 'userguide',
    #     'gallery_dirs': 'auto_userguide',
    #     'filename_pattern': r'.*\.py$',
    #     'within_subsection_order': FileNameSortKey,
    # }
    # ]
# regex to filter which files to include
    # Optional:
    # 'backreferences_dir': 'generated',
    # 'doc_module': ('your_module_name',),
    # 'image_scrapers': ('matplotlib',),
# }


language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"
# html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "github_url": "https://github.com/crispim-enc/Infinity-Tomographic-Reconstruction",
    "navbar_align": "content",  # optional: left | content | right
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_links": [
        {"name": "Installation", "url": "installation.html", "internal": True},
        {"name": "User Guide", "url": "gettingstarted.html", "internal": True},
        {"name": "Examples", "url": "auto_examples/index.html", "internal": True},
        {"name": "API Reference", "url": "modules.html", "internal": True},
    ]
}

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
    'custom.css',
]