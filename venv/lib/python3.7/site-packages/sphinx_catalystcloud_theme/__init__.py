import os

__version__ = '0.0.1'
__version_full__ = __version__


def get_html_theme_path():
    """Return list of HTML theme paths."""
    cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    return cur_dir


def setup(app):
    app.add_html_theme('sphinx_catalystcloud_theme', os.path.abspath(os.path.dirname(__file__)))
