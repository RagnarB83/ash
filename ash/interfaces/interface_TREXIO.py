import subprocess as sp
import os
import shutil
import numpy as np

from ash.functions.functions_general import ashexit

# Wrapper around TREXIO library
# https://github.com/TREX-CoE/trexio
# https://github.com/TREX-CoE/trexio_tools

def trexio_wrapper():
    try:
        import trexio
    except ImportError:
        print("Problem importing trexio library: See: https://github.com/TREX-CoE/trexio")
        print("Example: conda install -c conda-forge trexio")
        ashexit()
