import os
import sys
from sphinx_gallery.sorting import FileNameSortKey

sys.path.insert(0, os.path.abspath("../../src"))

project = "MFML-4-QC"
copyright = "2026, Vivin Vinod"
author = "Vivin Vinod"
# UPDATE 1: Bump the release version to match your new package version
release = "1.5.0"

# html_title = "MFML-4-QC Documentation"

extensions = [
    "sphinx.ext.autodoc",  # Automatically pulls docstrings from classes/functions
    "sphinx.ext.napoleon",  # Allows Sphinx to read docstrings
    "sphinx.ext.viewcode",  # Adds a "[source]" button next to documentation to view the raw Python code
    "sphinx.ext.mathjax",  # Latex math rendering
    "sphinx_gallery.gen_gallery",  # Turns examples/ folder into tutorials
    "nbsphinx",  # Render Jupyter notebooks
    "sphinx.ext.autosummary",
]
# autosummary automatically generate the .rst files
autosummary_generate = True

# Configure Sphinx Gallery
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "within_subsection_order": FileNameSortKey,
    "filename_pattern": r"\.py$",
    "show_memory": True,
}

# UPDATE 2: Force Jupyter notebooks to re-execute every time you build
nbsphinx_execute = "always"

html_theme = "pydata_sphinx_theme"
