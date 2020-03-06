.. _sphinx_rtd_theme: https://github.com/snide/sphinx_rtd_theme
.. _syncing a fork: https://help.github.com/articles/syncing-a-fork/
.. _bower: http://www.bower.io
.. _sphinx: http://www.sphinx-doc.org
.. _compass: http://www.compass-style.org
.. _sass: http://www.sass-lang.com
.. _wyrm: http://www.github.com/snide/wyrm/
.. _grunt: http://www.gruntjs.com
.. _node: http://www.nodejs.com
.. _demo: http://docs.readthedocs.org
.. _hidden: http://sphinx-doc.org/markup/toctree.html

************************
RealTimeCat Sphinx Theme
************************

.. contents:: 

RealTimeCat Sphinx theme is a fork of the excellent sphinx_rtd_theme_. See the documentation there for general information.

Stay in sync
============

In order to always stay in sync with the upstream sphinx_rtd_theme_, this repository contains two remotes: ``upstream`` and ``origin``.
See Github's `syncing a fork`_ article for more information.


Installation
============

Via package
-----------

Download the package or add it to your ``requirements.txt`` file:

.. code:: bash

    $ pip install rtcat_sphinx_theme

In your ``conf.py`` file:

.. code:: python

    import rtcat_sphinx_theme

    html_theme = "rtcat_sphinx_theme"

    html_theme_path = [rtcat_sphinx_theme.get_html_theme_path()]

Via git or download
-------------------

Symlink or submodule/subtree the ``RTCat/rtcat_sphinx_theme`` repository into your documentation at
``docs/_themes/rtcat_sphinx_theme`` then add the following two settings to your Sphinx
conf.py file:

.. code:: python

    html_theme = "rtcat_sphinx_theme"
    html_theme_path = ["_themes/rtcat_sphinx_theme", ]

Editing
=======

The rtcat_sphinx_theme is primarily a sass_ project that requires a few other sass libraries. I'm
using bower_ to manage these dependencies and sass_ to build the css. The good news is
I have a very nice set of grunt_ operations that will not only load these dependencies, but watch
for changes, rebuild the sphinx demo docs and build a distributable version of the theme.
The bad news is this means you'll need to set up your environment similar to that
of a front-end developer (vs. that of a python developer). That means installing node and ruby.

Set up your environment
-----------------------

1. Install sphinx_ into a virtual environment.

.. code::

    virtualenv ve
    source ve/bin/activate
    pip install sphinx

2. Install sass

.. code::

    gem install sass

2. Install node, bower and grunt.

.. code::

    // Install bower and grunt
    npm install -g bower grunt-cli

    // Now that everything is installed, let's install the theme dependecies.
    npm install

Now that our environment is set up, make sure you're in your virtual environment, go to
this repository in your terminal and run grunt:

.. code::

    grunt

This default task will do the following **very cool things that make it worth the trouble**.

1. It'll install and update any bower dependencies.
2. It'll run sphinx and build new docs.
3. It'll watch for changes to the sass files and build css from the changes.
4. It'll rebuild the sphinx docs anytime it notices a change to .rst, .html, .js
   or .css files.


