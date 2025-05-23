import subprocess as sp
import os
import shutil
import numpy as np

from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader,listdiff, pygrep
import ash.settings_ash
from ash.modules.module_coords import elemstonuccharges, write_xyzfile
from ash.modules.module_theory import Theory

# Basic interface to DFTD4: https://github.com/dftd4/dftd4
# This interface simply calls DFTD4 on an ASH-Fragment and returns
# the D4 dispersion energy and gradient

#Class in case we want to use it for WrapTheory:
# Use like this:   dftd4 = DFTD4Theory(functional="PBE")
# DFT_plus_D4 = WrapTheory(theory1=orca, theory2=dftd4)
class DFTD4Theory:
    def __init__(self, functional=None, printlevel=2, numcores=1):
        super().__init__()
        if functional is None:
            print(BC.FAIL, "No functional keyword provided. Exiting.", BC.END)
            ashexit()
        self.functional = functional
        self.theorynamelabel="DFTD4"
        self.printlevel=printlevel

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):

        frag = ash.Fragment(elems=elems, coords=current_coords,printlevel=0)
        eg = calc_DFTD4(fragment=frag, functional=self.functional, Grad=Grad, printlevel=self.printlevel)


        if Grad:
            self.energy = eg[0]
            self.gradient=eg[1]
        else:
            self.energy=eg

        if Grad is True:
            return self.energy, self.gradient
        else:
            return self.energy



# Simple standalone function to get DFTD4 energy and gradient for a fragment and functional-keyword
def calc_DFTD4(fragment=None, functional=None, Grad=True, printlevel=2):

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
    if Grad:
        if printlevel > 2:
            print("DFT-D4 gradient:", gradient)
        return energy, gradient
    else:
        return energy


# Simple standalone function to get gcp energy and gradient for a fragment and functional-keyword
def calc_gcp(fragment=None, xyzfile=None, current_coords=None, elems=None, functional=None, Grad=True, printlevel=2):

    print("calc_gcp: interface to gCP")
    if shutil.which('mctc-gcp') is None:
        print("Problem finding mctc-gcp binary in PATH. Make sure you have installed conda-forge package gcp-correction in this environment: See: https://github.com/grimme-lab/gcp")
        print("Example: mamba install gcp-correction")
        ashexit()

    # Early exits
    if fragment is None:
        print(BC.FAIL, "No fragment provided. Exiting.", BC.END)
        ashexit()

    # Early exits
    if functional is None:
        print(BC.FAIL, "No functional keyword provided. Exiting.", BC.END)
        ashexit()

    # Write fragment XYZ-file
    # NOTE: For QM/MM write QM-region XYZ-file first or pass coords and elems
    if fragment is not None:
        fragment.write_xyzfile(xyzfilename="gcpgeo.xyz", writemode='w')
        xyzfile="gcpgeo.xyz"
        numatoms=fragment.numatoms
    elif xyzfile is not None:
        with open ("gcpgeo.xyz") as fh: numatoms=int(next(fh))

    elif current_coords is not None:
        write_xyzfile(elems, current_coords, "gcpgeo.xyz", printlevel=2, writemode='w', title="title")
        xyzfile="gcpgeo.xyz"
        numatoms=len(current_coords)

    command_list=['mctc-gcp', xyzfile, '-l', functional]
    if Grad:
        command_list.append('--grad')

    print("command_list:", command_list)
    with open('gcp.out', 'w') as ofile:
        sp.run(command_list, env=os.environ, stdout=ofile, stderr=ofile)

    result = pygrep("Egcp","gcp.out")

    energy = float(result[1])
    print("gcp energy:", energy)
    if Grad:
        gradient = gcpgradientgrab(numatoms,"gcp_gradient")
        if printlevel > 2:
            print("gcp gradient:", gradient)
        return energy, gradient
    else:
        return energy

# Grab gcp gradient (Eh/Bohr) from  file
def gcpgradientgrab(numatoms,file):
    gradient = np.zeros((numatoms, 3))
    with open(file) as f:
        for count,line in enumerate(f):
            val_x=float(line.split()[0].replace("D","E"))
            val_y = float(line.split()[1].replace("D","E"))
            val_z = float(line.split()[2].replace("D","E"))
            gradient[count] = [val_x,val_y,val_z]
    return gradient

class gcpTheory(Theory):
    def __init__(self, functional=None, printlevel=2, numcores=1):
        super().__init__()
        if functional is None:
            print(BC.FAIL, "No functional keyword provided. Exiting.", BC.END)
            ashexit()
        self.functional = functional
        self.theorynamelabel="gCP"
        self.printlevel=printlevel

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):

        frag = ash.Fragment(elems=elems, coords=current_coords,printlevel=0)
        eg = calc_gcp(fragment=frag, functional=self.functional, Grad=Grad, printlevel=self.printlevel)
        if Grad:
            self.energy = eg[0]
            self.gradient=eg[1]
        else:
            self.energy=eg

        if Grad is True:
            return self.energy, self.gradient
        else:
            return self.energy
