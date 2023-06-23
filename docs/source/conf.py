# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GPErks'
copyright = '2023, Stefano Longobardi, Gianvito Taneburgo'
author = 'Stefano Longobardi, Gianvito Taneburgo'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'numpydoc',
   # 'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_material'
html_static_path = ['_static']

# html_logo = 'GPErks_logo.png'

html_theme_options = {
    'repo_url': 'https://github.com/stelong/GPErks/',
    'repo_name': 'GPErks',
    'html_minify': True,
    'css_minify': True,
    'nav_title': 'GPErks: A python library to (bene)fit Gaussian Process Emulators',
    'logo_icon': '&#xe88a',
    'globaltoc_depth': 2
}