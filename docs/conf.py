"""Configuration for the Sphinx documentation builder."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Ensure the project package is importable
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

project = "SimulationSupport"
author = "SXS collaboration"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build"]

html_theme = "furo"
html_static_path: list[str] = ["_static"]

# Keep type hints in the signature for readability
autodoc_typehints = "description"

# Render module-level __all__ by default
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
