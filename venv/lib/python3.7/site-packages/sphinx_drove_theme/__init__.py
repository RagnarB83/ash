"""Sphinx drove theme.

From https://github.com/droveio/sphinx-drove-theme.

"""
import os

VERSION = (1, 11, 0)

__version__ = ".".join(str(v) for v in VERSION)
__version_full__ = __version__


def get_html_theme_path():
    """Return list of HTML theme paths."""
    cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    return cur_dir
