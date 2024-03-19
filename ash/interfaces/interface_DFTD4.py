import subprocess as sp
import os
import shutil
import numpy as np

from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader,listdiff
import ash.settings_ash
from ash.modules.module_coords import elemstonuccharges

# Basic interface to DFTD4: https://github.com/dftd4/dftd4
# This interface simply calls DFTD4 on an ASH-Fragment and returns
# the D4 dispersion energy and gradient

#NOTE: we can't currently use this for opt,freq and md jobs as this would have to be called in each step.

#Class in case we want to use it for WrapTheory:
# Use like this:   dftd4 = DFTD4Theory(functional="PBE")
# DFT_plus_D4 = WrapTheory(theory1=orca, theory2=dftd4)
class DFTD4Theory:
    def __init__(self, functional=None, printlevel=2, numcores=1):

        if functional is None:
            print(BC.FAIL, "No functional keyword provided. Exiting.", BC.END)
            ashexit()
        self.functional = functional

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):

        frag = ash.Fragment(elems=elems, coords=current_coords,printlevel=0)
        energy, gradient = calc_DFTD4(fragment=frag, functional=self.functional, Grad=Grad)


        if Grad is True:
            return energy, gradient
        else:
            return energy


# Simple standalone function to get DFTD4 energy and gradient for a fragment and functional-keyword
def calc_DFTD4(fragment=None, functional=None, Grad=True):

    try:
        from dftd4.interface import Structure, DampingParam, DispersionModel
    except ImportError:
        print("Problem importing dftd4 python package. Make sure you have installed dftd4-python (and dftd4) in this environment: See: https://github.com/dftd4/dftd4")
        print("Example: conda install dftd4-python")
        ashexit()

    # Early exits
    if fragment is None:
        print(BC.FAIL, "No fragment provided. Exiting.", BC.END)
        ashexit()

    # Early exits
    if functional is None:
        print(BC.FAIL, "No functional keyword provided. Exiting.", BC.END)
        ashexit()

    # Converting coords to bohr and element symbols to atomic numbers
    numbers = np.array(elemstonuccharges(fragment.elems))
    positions=fragment.coords*ash.constants.ang2bohr

    # Create model and calc energy and gradient
    model = DispersionModel(numbers, positions)
    res = model.get_dispersion(DampingParam(method=functional), grad=Grad)
    energy = res.get("energy")
    gradient = res.get("gradient")

    print("DFT-D4 energy:", energy)
    print("DFT-D4 gradient:", gradient)

    return energy, gradient
