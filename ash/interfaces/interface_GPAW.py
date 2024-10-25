import time

from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader,listdiff
import ash.modules.module_coords
from ash.modules.module_results import ASH_Results
from ash.modules.module_theory import QMTheory
from ash.functions.functions_elstructure import get_ec_entropy,get_entropy
import os
import sys
import glob
import numpy as np
from functools import reduce
import random
import copy


# TODO: Brillouin zone sampling
# TODO: symmetry
# TODO: occupation number smearing
# TODO convergence

class GPAWTheory(QMTheory):
    def __init__(self, printsetting=False, printlevel=2, numcores=1, label="gpaw", filename="gpaw", functional=None,
                 basis=None, nbands=None, gridpoints=None, mode=None):

        self.theorynamelabel="GPAW"
        self.theorytype="QM"
        self.analytic_hessian=True
        self.printlevel=printlevel
        self.label=label
        self.numcores=numcores

        print_line_with_mainheader(f"{self.theorynamelabel} initialization")

        if functional is None:
            print("No functional provided. Example: functional=\"PBE\" Exiting")
            ashexit()
        if nbands is None:
            print("No nbands provided. Example: nbands=2, Exiting")
            ashexit()
        if gridpoints is None:
            print("No gridpoints provided. Example: gridpoints=(24,24,24) Exiting")
            ashexit()
        if basis is None:
            print("No basis provided. Exiting")
            ashexit()
        if mode is None:
            print("No mode provided. Example: mode=\"fd\" or mode=\"PW(200)\" or mode=\"lcao\"  Exiting")
            ashexit()
        self.functional=functional
        self.nbands=nbands
        self.gridpoints=gridpoints
        self.mode=mode
        self.basis=basis
        self.filename=filename
        # Basis can be name-string or a dict: e.g. basis={'H': 'sz', 'C': 'dz', 7: 'dzp'}

    # Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
        print("Setting numcores to: ", self.numcores)

    # Cleanup after run.
    def cleanup(self):
        files=['timer.dat', self.filename+'.dat',self.filename+'.chk' ]
        print(f"Cleaning up old {self.theorynamelabel} files")
        for f in files:
            print("Removing file:", f)
            try:
                os.remove(f)
            except:
                print("Error removing file", f)

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
            charge=None, mult=None):

        module_init_time=time.time()
        if self.printlevel >0:
            print(BC.OKBLUE,BC.BOLD, f"------------PREPARING {self.theorynamelabel} INTERFACE-------------", BC.END)
            print("Object-label:", self.label)
            print("Run-label:", label)

        # Load gpaw and ase
        from gpaw import GPAW, PW
        from ase import Atoms

        if mult > 1:
            spinpol = True
        else:
            #TODO: add option to do spinpol True even if singlet
            spinpol = False

        # Creating ASE atoms object
        print("Creating ASE atoms object")
        atoms = Atoms(elems,positions=current_coords)

        calc = GPAW(mode=self.mode, nbands=self.nbands, xc=self.functional, 
                    gpts=self.gridpoints, basis=self.basis, charge=charge, spinpol=spinpol,
                    txt=self.filename)
        atoms.calc = calc


        if Grad is True:
            forces = atoms.get_forces()
            print("forces:", forces)
        else:
            self.energy = atoms.get_potential_energy()
        if self.printlevel >= 1:
            print()
            print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        if Grad is True:
            if self.printlevel >=0:
                print(f"Single-point {self.theorynamelabel} energy:", self.energy)
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            if PC is True:
                return self.energy, self.gradient, self.pcgrad
            else:
                return self.energy, self.gradient
        else:
            if self.printlevel >=0:
                print(f"Single-point {self.theorynamelabel} energy:", self.energy)
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            return self.energy
