# ASH - A GENERAL COMPCHEM AND QM/MM ENVIRONMENT

#Python libraries
import os
import shutil
import numpy as np
import copy
import subprocess as sp
import glob
import sys
import inspect
import time

#ASH modules
import ash
from functions_general import blankline,BC,listdiff,print_time_rel,print_time_rel_and_tot,pygrep,printdebug,read_intlist_from_file
# Fragment class and coordinate functions
import module_coords
from module_coords import get_molecules_from_trajectory,eldict_covrad,write_pdbfile
#Parallel 
import functions_parallel
from functions_parallel import Singlepoint_parallel,run_QMMM_SP_in_parallel
#Freq
from module_freq import AnFreq,NumFreq,approximate_full_Hessian_from_smaller,calc_rotational_constants
#Constants
import constants
#functions related to electronic structure
import functions_elstructure
#Spinprojection
from module_spinprojection import SpinProjectionTheory
#Surface
from module_surface import calc_surface,calc_surface_fromXYZ,read_surfacedict_from_file
import settings_ash
from ash_header import print_ash_header
#QMcode interfaces
from interface_ORCA import ORCATheory
from interface_Psi4 import Psi4Theory
from interface_dalton import DaltonTheory
from interface_pyscf import PySCFTheory
from interface_MRCC import MRCCTheory
from interface_CFour import CFourTheory
from interface_xtb import xTBTheory
#MM: external and internal
from interface_OpenMM import OpenMMTheory
from module_MM import NonBondedTheory,UFFdict,UFF_modH_dict,LJCoulpy,coulombcharge,LennardJones,LJCoulombv2,LJCoulomb,MMforcefield_read
#QM/MM
from module_QMMM import QMMMTheory
from module_polembed import PolEmbedTheory
#Solvation
#NOTE: module_solvation.py or module_solvation2.py To be cleaned up
import functions_solv
#Molcrys
import module_molcrys
from module_molcrys import *
# Geometry optimization
from functions_optimization import SimpleOpt,BernyOpt
from interface_geometric import geomeTRICOptimizer
#Workflows, benchmarking etc
import module_workflows
import module_highlevel_workflows
from module_workflows import ReactionEnergy,thermochemprotocol_reaction,thermochemprotocol_single,confsampler_protocol
import module_benchmarking
from module_benchmarking import run_benchmark
#Other interfaces
import interface_crest


#Julia dependency
#Current behaviour: We try to import, if not possible then we continue
load_julia = True
if load_julia is True:
    try:
        print("Import PyJulia interface")
        from julia.api import Julia
        from julia import Main
        #Hungarian package needs to be installed
        try:
            from julia import Hungarian
        except:
            print("Problem loading Julia packages: Hungarian")
        ashpath = os.path.dirname(ash.__file__)
        #Various Julia functions
        print("Loading Julia functions")
        Main.include(ashpath + "/functions_julia.jl")
    except:
        print("Problem importing Pyjulia")
        print("Make sure Julia is installed, PyJulia within Python, Pycall within Julia, Julia packages have been installed and you are using python-jl")
        print("Python routines will be used instead when possible")
        #TODO: We should here set a variable that would pick py version of routines instead

#############################################################

#Single-point energy function
#NOTE: 
def Singlepoint(fragment=None, theory=None, Grad=False):
    print("")
    '''
    The Singlepoint function carries out a single-point energy calculation
    :param fragment:
    :type fragment: ASH object of class Fragment
    :param theory:
    :type theory: ASH theory object
    :param Grad: whether to do Gradient or not.
    :type Grad: Boolean.
    '''
    if fragment is None or theory is None:
        print(BC.FAIL,"Singlepoint requires a fragment and a theory object",BC.END)
        exit(1)
    coords=fragment.coords
    elems=fragment.elems
    # Run a single-point energy job with gradient
    if Grad ==True:
        print(BC.WARNING,"Doing single-point Energy+Gradient job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
        # An Energy+Gradient calculation where we change the number of cores to 12
        energy,gradient= theory.run(current_coords=coords, elems=elems, Grad=True)
        print("Energy: ", energy)
        return energy,gradient
    # Run a single-point energy job without gradient (default)
    else:
        print(BC.WARNING,"Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)

        energy = theory.run(current_coords=coords, elems=elems)
        print("Energy: ", energy)
        #Now adding total energy to fragment
        fragment.energy=energy
        return energy

#Theory classes

# Theory object that always gives zero energy and zero gradient. Useful for setting constraints
class ZeroTheory:
    def __init__(self, fragment=None, charge=None, mult=None, printlevel=None, nprocs=1, label=None):
        self.nprocs=nprocs
        self.charge=charge
        self.mult=mult
        self.printlevel=printlevel
        self.label=label
        self.fragment=fragment
        pass
    def run(self, current_coords=None, elems=None, Grad=False, PC=False, nprocs=None ):
        self.energy = 0.0
        #Numpy object
        self.gradient = np.zeros((len(elems), 3))
        return self.energy,self.gradient


settings_ash.init()
