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

#Add local geometric dir to syspath
sys.path.insert(0, ashpath+"/geometric-master")


from ash.functions.functions_general import blankline, BC, listdiff, print_time_rel, print_time_rel_and_tot, pygrep, \
    printdebug, read_intlist_from_file, frange, writelisttofile, load_julia_interface, read_datafile, write_datafile, ashexit

#Results dataclass 
from ash.modules.module_results import ASH_Results

# Fragment class and coordinate functions
import ash.modules.module_coords
from ash.modules.module_coords import get_molecules_from_trajectory, eldict_covrad, write_pdbfile, Fragment, read_xyzfile, \
    write_xyzfile, make_cluster_from_box, read_ambercoordinates, read_gromacsfile, split_multimolxyzfile
from ash.modules.module_coords import remove_atoms_from_system_CHARMM, add_atoms_to_system_CHARMM, getwaterconstraintslist,\
    QMregionfragexpand, read_xyzfiles, Reaction, define_XH_constraints, simple_get_water_constraints

# Singlepoint
import ash.modules.module_singlepoint
from ash.modules.module_singlepoint import Singlepoint, newSinglepoint, ZeroTheory, ScriptTheory, Singlepoint_fragments,\
     Singlepoint_theories, Singlepoint_fragments_and_theories, Singlepoint_reaction

# Parallel
import ash.functions.functions_parallel
from ash.functions.functions_parallel import Job_parallel, Simple_parallel

Singlepoint_parallel = Job_parallel

# Freq
from ash.modules.module_freq import AnFreq, NumFreq, approximate_full_Hessian_from_smaller, calc_rotational_constants,\
    get_dominant_atoms_in_mode, write_normalmode,calc_hessian_xtb, wigner_distribution,write_hessian,read_hessian

# Constants
import ash.constants

# functions related to electronic structure
import ash.functions.functions_elstructure
from ash.functions.functions_elstructure import read_cube, write_cube_diff, \
    NOCV_density_ORCA, difference_density_ORCA, NOCV_Multiwfn,write_cube_sum,write_cube_product,create_density_from_orb

#multiwfn interface
import ash.interfaces.interface_multiwfn
from ash.interfaces.interface_multiwfn import multiwfn_run
# Spinprojection
from ash.modules.module_spinprojection import SpinProjectionTheory
#DualTheory
from ash.modules.module_dualtheory import DualTheory

# Surface
from ash.modules.module_surface import calc_surface, calc_surface_fromXYZ, read_surfacedict_from_file, \
    write_surfacedict_to_file

# QMcode interfaces
from ash.interfaces.interface_ORCA import ORCATheory, counterpoise_calculation_ORCA, ORCA_External_Optimizer, run_orca_plot, MolecularOrbitalGrab, \
        run_orca_mapspc, make_molden_file_ORCA, grab_coordinates_from_ORCA_output, ICE_WF_CFG_CI_size, orca_frag_guess, orblocfind, ORCAfinalenergygrab
import ash.interfaces.interface_ORCA

from ash.interfaces.interface_Psi4 import Psi4Theory
from ash.interfaces.interface_dalton import DaltonTheory
from ash.interfaces.interface_pyscf import PySCFTheory
from ash.interfaces.interface_ipie import ipieTheory
from ash.interfaces.interface_dice import DiceTheory
from ash.interfaces.interface_block import BlockTheory
from ash.interfaces.interface_MRCC import MRCCTheory
from ash.interfaces.interface_QUICK import QUICKTheory
from ash.interfaces.interface_TeraChem import TeraChemTheory
from ash.interfaces.interface_sparrow import SparrowTheory
from ash.interfaces.interface_NWChem import NWChemTheory
from ash.interfaces.interface_CP2K import CP2KTheory
from ash.interfaces.interface_BigDFT import BigDFTTheory
from ash.interfaces.interface_deMon import deMon2kTheory

from ash.interfaces.interface_CFour import CFourTheory
from ash.interfaces.interface_xtb import xTBTheory
from ash.interfaces.interface_PyMBE import PyMBETheory

# MM: external and internal
from ash.interfaces.interface_OpenMM import OpenMMTheory, OpenMM_MD, OpenMM_MDclass, OpenMM_Opt, OpenMM_Modeller, \
     OpenMM_box_relaxation, write_nonbonded_FF_for_ligand, solvate_small_molecule, \
        OpenMM_metadynamics, Gentle_warm_up_MD, check_gradient_for_bad_atoms, get_free_energy_from_biasfiles, \
        free_energy_from_bias_array,metadynamics_plot_data
from ash.modules.module_MM import NonBondedTheory, UFFdict, UFF_modH_dict, LJCoulpy, coulombcharge, LennardJones, \
    LJCoulombv2, LJCoulomb, MMforcefield_read
#MDtraj
from ash.interfaces.interface_mdtraj import MDtraj_imagetraj, MDtraj_slice, MDtraj_RMSF, MDtraj_coord_analyze

# QM/MM
from ash.modules.module_QMMM import QMMMTheory, actregiondefine, read_charges_from_psf
from ash.modules.module_polembed import PolEmbedTheory

# Knarr
from ash.interfaces.interface_knarr import NEB, NEBTS

# ASE-Dynamics
from ash.interfaces.interface_ASE import Dynamics_ASE

# Plumed interface
from ash.interfaces.interface_plumed import plumed_ASH, MTD_analyze

# Solvation
# NOTE: module_solvation.py or module_solvation2.py To be cleaned up
import ash.functions.functions_solv

# Molcrys
import ash.modules.module_molcrys
from ash.modules.module_molcrys import molcrys, Fragmenttype

# Geometry optimization
from ash.functions.functions_optimization import SimpleOpt, BernyOpt
#TODO: Delete eventually:
from ash.interfaces.interface_geometric import oldgeomeTRICOptimizer, oldGeomeTRICOptimizerClass
#NEW version. Will replace other one
from ash.interfaces.interface_geometric_new import geomeTRICOptimizer,GeomeTRICOptimizerClass
Optimizer = geomeTRICOptimizer
Opt = geomeTRICOptimizer

# PES
from ash.modules.module_PES_rewrite import PhotoElectron, potential_adjustor_DFT, plot_PES_Spectrum,Read_old_PES_results

# Workflows, benchmarking etc
import ash.modules.module_workflows
import ash.modules.module_highlevel_workflows
from ash.modules.module_highlevel_workflows import ORCA_CC_CBS_Theory, Reaction_FCI_Analysis, make_ICE_theory

CC_CBS_Theory = ORCA_CC_CBS_Theory #TODO: Temporary alias

from ash.modules.module_workflows import ReactionEnergy, thermochemprotocol_reaction, thermochemprotocol_single, \
    confsampler_protocol, auto_active_space, calc_xyzfiles, ProjectResults, Reaction_Highlevel_Analysis, FormationEnthalpy, \
    AutoNonAufbau, ExcitedStateSCFOptimizer,TDDFT_vib_ave
import ash.modules.module_benchmarking
from ash.modules.module_benchmarking import run_benchmark


#Plotting
import ash.modules.module_plotting
from ash.modules.module_plotting import reactionprofile_plot, contourplot, plot_Spectrum, MOplot_vertical, ASH_plot

# Other
import ash.interfaces.interface_crest
from ash.interfaces.interface_crest import call_crest, call_crest_entropy, get_crest_conformers

# Initialize settings
import ash.settings_ash

# Print header
import ash.ash_header
ash.ash_header.print_header()

# Exit command (footer)
if ash.settings_ash.settings_dict["print_exit_footer"] is True:
    atexit.register(ash.ash_header.print_footer)
    if ash.settings_ash.settings_dict["print_full_timings"] is True:
        atexit.register(ash.ash_header.print_timings)

# Julia dependency. Load in the beginning or not. 
#As both PyJulia and PythonCall are a bit slow to load, it is best to only load when needed (current behaviour)
if ash.settings_ash.settings_dict["load_julia"] is True:
    try:
        print("Importing Julia interface and loading functions")
        Juliafunctions = load_julia_interface()

    except ImportError:
        print("Problem importing Julia interface")
        print(
            "Make sure Julia is installed, Pythoncall/juliacall and the required Julia packages have been installed.")
        print("Proceeding. Slower Python routines will used instead when possible")
        # Connectivity code in Fragment
        ash.settings_ash.settings_dict["connectivity_code"] = "py"
        # LJ+Coulomb and pairpot arrays in nonbonded MM
        ash.settings_ash.settings_dict["nonbondedMM_code"] = "py"
