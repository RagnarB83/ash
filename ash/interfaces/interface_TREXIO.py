import subprocess as sp
import os
import shutil
import numpy as np

from ash.functions.functions_general import ashexit

# Wrapper around TREXIO library
# https://github.com/TREX-CoE/trexio
# https://github.com/TREX-CoE/trexio_tools

# In progress
# Need to add properties
# Support ORCA and PySCF writers

# TODO: add properties
def read_trexio_file(filename="trexio", back_end_type="text"):
    try:
        import trexio
    except ImportError:
        print("Problem importing trexio library: See: https://github.com/TREX-CoE/trexio")
        print("Example: conda install -c conda-forge trexio")
        ashexit()
    if back_end_type.lower() == "text":
        back_end=trexio.TREXIO_TEXT
    elif back_end_type.lower() == "hdf5":
        back_end=trexio.TREXIO_HDF5
    else:
        print("Unknown format")
        ashexit()

    open_file = trexio.File(filename, mode='r', back_end=back_end)

    charges_r = trexio.read_nucleus_charge(open_file)
    labels_r = trexio.read_nucleus_label(open_file)
    coords_r = trexio.read_nucleus_coord(open_file)

    open_file.close()

    return charges_r, coords_r, labels_r

# Very basic trexio file writer, only molecule info for the moment
def write_trexio_file(fragment, filename="trexio", back_end_type="text"):
    try:
        import trexio
    except ImportError:
        print("Problem importing trexio library: See: https://github.com/TREX-CoE/trexio")
        print("Example: conda install -c conda-forge trexio")
        ashexit()
    if back_end_type.lower() == "text":
        back_end=trexio.TREXIO_TEXT
        filename=filename+".text"
    elif back_end_type.lower() == "hdf5":
        back_end=trexio.TREXIO_HDF5
        filename=filename+".h5"
    else:
        print("Unknown format")
        ashexit()
    print("filename:", filename)
    print("Backend :", back_end)

    open_file = trexio.File(filename, mode='w', back_end=back_end)

    print("Writing nucleus sum (numatoms):", fragment.numatoms)
    trexio.write_nucleus_num(open_file, fragment.numatoms)

    # Write charges and coordinates
    print("Writing nuclear charges, labels and coordinates")
    trexio.write_nucleus_label(open_file, fragment.elems)
    trexio.write_nucleus_charge(open_file, fragment.nuc_charges)
    trexio.write_nucleus_coord(open_file, fragment.coords)

    open_file.close()

    print("Wrote trexio file:", filename)