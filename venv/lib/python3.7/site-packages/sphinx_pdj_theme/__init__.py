import os

VERSION = '0.1.6'


def get_html_theme_path():
    """Return list of HTML theme paths."""
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    return cur_dir


def setup(app):
    app.add_html_theme('sphinx_pdj_theme', get_html_theme_path())
