"""
ASH - A MULTISCALE MODELLING PROGRAM

"""
# Python libraries
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
# ASH modules
###############
import ash

# Adding modules,interfaces directories to sys.path
ashpath = os.path.dirname(ash.__file__)
# sys.path.insert(1, ashpath+'/modules')
# sys.path.insert(1, ashpath+'/interfaces')
# sys.path.insert(1, ashpath+'/functions')

from functions.functions_general import blankline, BC, listdiff, print_time_rel, print_time_rel_and_tot, pygrep, \
    printdebug, read_intlist_from_file, frange, writelisttofile, load_julia_interface

# Fragment class and coordinate functions
import modules.module_coords
from modules.module_coords import get_molecules_from_trajectory, eldict_covrad, write_pdbfile, Fragment, read_xyzfile, \
    write_xyzfile, make_cluster_from_box, read_ambercoordinates, read_gromacsfile
from modules.module_coords import remove_atoms_from_system_CHARMM, add_atoms_to_system_CHARMM, getwaterconstraintslist,\
    QMregionfragexpand, read_xyzfiles

# Singlepoint
import modules.module_singlepoint
from modules.module_singlepoint import Singlepoint, newSinglepoint, ZeroTheory, Singlepoint_fragments, Singlepoint_theories, Singlepoint_fragments_and_theories

# Parallel
import functions.functions_parallel
from functions.functions_parallel import Singlepoint_parallel, run_QMMM_SP_in_parallel

# Freq
from modules.module_freq import AnFreq, NumFreq, approximate_full_Hessian_from_smaller, calc_rotational_constants,get_dominant_atoms_in_mode, write_normalmode

# Constants
import constants

# functions related to electronic structure
import functions.functions_elstructure

# Spinprojection
from modules.module_spinprojection import SpinProjectionTheory

# Surface
from modules.module_surface import calc_surface, calc_surface_fromXYZ, read_surfacedict_from_file, \
    write_surfacedict_to_file

# QMcode interfaces
from interfaces.interface_ORCA import ORCATheory, counterpoise_calculation_ORCA, ORCA_External_Optimizer, run_orca_plot
import interfaces.interface_ORCA

from interfaces.interface_Psi4 import Psi4Theory
from interfaces.interface_dalton import DaltonTheory
from interfaces.interface_pyscf import PySCFTheory
from interfaces.interface_MRCC import MRCCTheory
from interfaces.interface_CFour import CFourTheory
from interfaces.interface_xtb import xTBTheory

# MM: external and internal
from interfaces.interface_OpenMM import OpenMMTheory, OpenMM_MD, OpenMM_MDclass, OpenMM_Opt, OpenMM_Modeller, \
    MDtraj_imagetraj, solvate_small_molecule, MDAnalysis_transform, OpenMM_box_relaxation
from modules.module_MM import NonBondedTheory, UFFdict, UFF_modH_dict, LJCoulpy, coulombcharge, LennardJones, \
    LJCoulombv2, LJCoulomb, MMforcefield_read

# QM/MM
from modules.module_QMMM import QMMMTheory, actregiondefine
from modules.module_polembed import PolEmbedTheory

# Knarr
from interfaces.interface_knarr import NEB

# ASE-Dynamics
from interfaces.interface_ASE import Dynamics_ASE

# Plumed interface
from interfaces.interface_plumed import plumed_ASH, MTD_analyze

# Solvation
# NOTE: module_solvation.py or module_solvation2.py To be cleaned up
import functions.functions_solv

# Molcrys
import modules.module_molcrys
from modules.module_molcrys import molcrys, Fragmenttype

# Geometry optimization
from functions.functions_optimization import SimpleOpt, BernyOpt
from interfaces.interface_geometric import geomeTRICOptimizer

# PES
import modules.module_PES
from modules.module_PES import PhotoElectronSpectrum, potential_adjustor_DFT

# Workflows, benchmarking etc
import modules.module_workflows
import modules.module_highlevel_workflows
from modules.module_highlevel_workflows import CC_CBS_Theory
from modules.module_workflows import ReactionEnergy, thermochemprotocol_reaction, thermochemprotocol_single, \
    confsampler_protocol, auto_active_space, calc_xyzfiles, ProjectResults, Reaction_Highlevel_Analysis, FormationEnthalpy, AutoNonAufbau
import modules.module_benchmarking
from modules.module_benchmarking import run_benchmark


#Plotting
import modules.module_plotting
from modules.module_plotting import reactionprofile_plot, contourplot, plot_Spectrum, MOplot_vertical, ASH_plot

# Other
import interfaces.interface_crest
from interfaces.interface_crest import call_crest, get_crest_conformers

# Initialize settings
import settings_ash

# Print header
import ash_header

ash_header.print_header()

# Exit command (footer)
if settings_ash.settings_dict["print_exit_footer"] is True:
    atexit.register(ash_header.print_footer)
    if settings_ash.settings_dict["print_full_timings"] is True:
        atexit.register(ash_header.print_timings)

# Julia dependency. Load in the beginning or not
if settings_ash.settings_dict["load_julia"] is True:
    try:
        print("Import PyJulia interface and loading functions")
        Juliafunctions = load_julia_interface()
        # Hungarian package needs to be installed
        # try:
        #    from julia import Hungarian
        # except:
        #    print("Problem loading Julia packages: Hungarian")

    except ImportError:
        print("Problem importing Pyjulia")
        print(
            "Make sure Julia is installed, PyJulia within Python, Pycall within Julia, Julia packages have been "
            "installed and you are using python3_ash")
        print("The python3_ash executable is present in your ASH directory (do chmod +x python3_ash)")
        print("Proceeding. Slower Python routines will used instead when possible")
        # Connectivity code in Fragment
        settings_ash.settings_dict["connectivity_code"] = "py"
        # LJ+Coulomb and pairpot arrays in nonbonded MM
        settings_ash.settings_dict["nonbondedMM_code"] = "py"
