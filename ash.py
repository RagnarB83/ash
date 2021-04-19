
"""
ASH - A MULTISCALE MODELLING PROGRAM

"""
#Python libraries
import os
import shutil
import numpy as np
import copy
import subprocess as sp
import glob
import os
import sys
import inspect
import time
import atexit




###############
#ASH modules
###############
import ash
#Adding modules,interfaces directories to sys.path
ashpath = os.path.dirname(ash.__file__)
sys.path.insert(1, ashpath+'/modules')
sys.path.insert(1, ashpath+'/interfaces')
sys.path.insert(1, ashpath+'/functions')

from functions_general import blankline,BC,listdiff,print_time_rel,print_time_rel_and_tot,pygrep,printdebug,read_intlist_from_file,frange

# Fragment class and coordinate functions
import module_coords
from module_coords import get_molecules_from_trajectory,eldict_covrad,write_pdbfile,Fragment,read_xyzfile,write_xyzfile,make_cluster_from_box, read_ambercoordinates, read_gromacsfile

#Singlepoint
import module_singlepoint
from module_singlepoint import Singlepoint,ZeroTheory

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

#QMcode interfaces
from interface_ORCA import ORCATheory
import interface_ORCA

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
from module_molcrys import molcrys,Fragmenttype

# Geometry optimization
from functions_optimization import SimpleOpt,BernyOpt
from interface_geometric import geomeTRICOptimizer

#PES
import module_PES
from module_PES import PhotoElectronSpectrum

#Workflows, benchmarking etc
import module_workflows
import module_highlevel_workflows
from module_workflows import ReactionEnergy,thermochemprotocol_reaction,thermochemprotocol_single,confsampler_protocol
import module_benchmarking
from module_benchmarking import run_benchmark

#Other
import interface_crest
from interface_crest import call_crest, get_crest_conformers


# Initialize settings
import settings_ash

#Print header
import ash_header
ash_header.print_header()


#Exit command (footer)
if settings_ash.settings_dict["print_exit_footer"] == True:
    atexit.register(ash_header.print_footer)
    if settings_ash.settings_dict["print_full_timings"] == True:
        atexit.register(ash_header.print_timings)



#Julia dependency. Current behaviour: 
# Default: load_julia is True and we try to load and continue if unsuccessful
# 
if settings_ash.settings_dict["load_julia"] == True:
    try:
        print("Import PyJulia interface")
        from julia.api import Julia
        from julia import Main
        #Hungarian package needs to be installed
        try:
            from julia import Hungarian
        except:
            print("Problem loading Julia packages: Hungarian")
        
        #Various Julia functions
        print("Loading Julia functions")
        ashpath = os.path.dirname(ash.__file__)
        Main.include(ashpath + "/functions/functions_julia.jl")
    except:
        print("Problem importing Pyjulia")
        print("Make sure Julia is installed, PyJulia within Python, Pycall within Julia, Julia packages have been installed and you are using python-jl")
        print("Python routines will be used instead when possible")
        #Connectivity code in Fragment
        settings_ash.settings_dict["connectivity_code"] = "py"
        #LJ+Coulomb and pairpot arrays in nonbonded MM
        settings_ash.settings_dict["nonbondedMM_code"] = "py"
        #TODO: We should here set a variable that would pick py version of routines instead




