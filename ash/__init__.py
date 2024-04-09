"""
ASH - A MULTISCALE MODELLING PROGRAM
R. Bjornsson
"""
# Python libraries
import numpy as np
import sys
import atexit
import pathlib
#Getting ASH-path
ashpath = str(pathlib.Path(__file__).parent.resolve())
print("ashpath:", ashpath)
###############
# ASH modules
###############
#import ash
# Adding modules,interfaces directories to sys.path
sys.path.insert(0, ashpath)
#ashpath = os.path.dirname(ash.__file__)
print("Sys path:", sys.path)
#Add local geometric dir to syspath
#sys.path.insert(0, ashpath+"/geometric-master")

from .functions.functions_general import create_ash_env_file,blankline, BC, listdiff, print_time_rel, print_time_rel_and_tot, pygrep, \
    printdebug, read_intlist_from_file, frange, writelisttofile, load_julia_interface, read_datafile, write_datafile, ashexit, natural_sort

#Results dataclass
from .modules.module_results import ASH_Results

# Fragment class and coordinate functions
import ash.modules.module_coords
from .modules.module_coords import get_molecules_from_trajectory, eldict_covrad, write_pdbfile, Fragment, read_xyzfile, \
    write_xyzfile, make_cluster_from_box, read_ambercoordinates, read_gromacsfile, split_multimolxyzfile,distance_between_atoms, \
    angle_between_atoms, dihedral_between_atoms, pdb_to_smiles, xyz_to_pdb_with_connectivity, writepdb_with_connectivity, mol_to_pdb, sdf_to_pdb
from .modules.module_coords import remove_atoms_from_system_CHARMM, add_atoms_to_system_CHARMM, getwaterconstraintslist,\
    QMregionfragexpand, read_xyzfiles, Reaction, define_XH_constraints, simple_get_water_constraints, print_internal_coordinate_table,\
    flexible_align_pdb, flexible_align_xyz, flexible_align

# Singlepoint
import ash.modules.module_singlepoint
from .modules.module_singlepoint import Singlepoint, newSinglepoint, ZeroTheory, ScriptTheory, Singlepoint_fragments,\
     Singlepoint_theories, Singlepoint_fragments_and_theories, Singlepoint_reaction


# Parallel
import ash.functions.functions_parallel
from .functions.functions_parallel import Job_parallel, Simple_parallel

Singlepoint_parallel = Job_parallel

# Freq
from .modules.module_freq import AnFreq, NumFreq, approximate_full_Hessian_from_smaller, calc_rotational_constants,\
    get_dominant_atoms_in_mode, write_normalmode,calc_hessian_xtb, wigner_distribution,write_hessian,read_hessian

# Constants
import ash.constants

# functions related to electronic structure
import ash.functions.functions_elstructure
from .functions.functions_elstructure import read_cube, write_cube, write_cube_diff, diffdens_tool, create_cubefile_from_orbfile, diffdens_of_cubefiles,  \
    NOCV_density_ORCA, difference_density_ORCA, NOCV_Multiwfn,write_cube_sum,write_cube_product,create_density_from_orb, make_molden_file, \
    diagonalize_DM_AO, diagonalize_DM, DM_AO_to_MO, DM_AO_to_MO, DM_MO_to_AO, select_space_from_occupations,select_indices_from_occupations

#multiwfn interface
import ash.interfaces.interface_multiwfn
from .interfaces.interface_multiwfn import multiwfn_run
# Spinprojection
from .modules.module_spinprojection import SpinProjectionTheory
#DualTheory and WrapTheory
from .modules.module_dualtheory import DualTheory,WrapTheory

# Surface
from .modules.module_surface import calc_surface, calc_surface_fromXYZ, read_surfacedict_from_file, write_surfacedict_to_file

# # QMcode interfaces
from .interfaces.interface_ORCA import ORCATheory, counterpoise_calculation_ORCA, ORCA_External_Optimizer, run_orca_plot, MolecularOrbitalGrab, \
    run_orca_mapspc, make_molden_file_ORCA, grab_coordinates_from_ORCA_output, ICE_WF_CFG_CI_size, orca_frag_guess, orblocfind, ORCAfinalenergygrab, \
    read_ORCA_json_file, create_ORCA_json_file,get_densities_from_ORCA_json,grab_ORCA_wfn,ORCA_orbital_setup
import ash.interfaces.interface_ORCA

from .interfaces.interface_Psi4 import Psi4Theory
from .interfaces.interface_dalton import DaltonTheory
from .interfaces.interface_pyscf import PySCFTheory, pyscf_MR_correction, pyscf_CCSD_T_natorb_selection,KS_inversion_kspies,DFA_error_analysis,pySCF_write_Moldenfile
density_potential_inversion=KS_inversion_kspies #Temporary
from .interfaces.interface_ipie import ipieTheory
from .interfaces.interface_dice import DiceTheory
from .interfaces.interface_block import BlockTheory
from .interfaces.interface_MRCC import MRCCTheory, run_MRCC_HLC_correction, convert_MRCC_Molden_file
from .interfaces.interface_QUICK import QUICKTheory
from .interfaces.interface_TeraChem import TeraChemTheory
from .interfaces.interface_sparrow import SparrowTheory
from .interfaces.interface_NWChem import NWChemTheory
from .interfaces.interface_Gaussian import GaussianTheory
from .interfaces.interface_CP2K import CP2KTheory
from .interfaces.interface_BigDFT import BigDFTTheory
from .interfaces.interface_deMon import deMon2kTheory
from .interfaces.interface_ccpy import ccpyTheory
from .interfaces.interface_MNDO import MNDOTheory

from .interfaces.interface_CFour import CFourTheory, run_CFour_HLC_correction, run_CFour_DBOC_correction, convert_CFour_Molden_file
from .interfaces.interface_xtb import xTBTheory
from .interfaces.interface_PyMBE import PyMBETheory
from .interfaces.interface_MLatom import MLatomTheory
from .interfaces.interface_DRACO import get_draco_radii
from .interfaces.interface_DFTD4 import DFTD4Theory, calc_DFTD4

# MM: external and internal
from .interfaces.interface_OpenMM import OpenMMTheory, OpenMM_MD, OpenMM_MDclass, OpenMM_Opt, OpenMM_Modeller, \
     OpenMM_box_equilibration, write_nonbonded_FF_for_ligand, solvate_small_molecule, small_molecule_parameterizer, \
        OpenMM_metadynamics, Gentle_warm_up_MD, check_gradient_for_bad_atoms, get_free_energy_from_biasfiles, \
        free_energy_from_bias_array,metadynamics_plot_data, merge_pdb_files

# General aliases
MolecularDynamics = OpenMM_MD
MetaDynamics = OpenMM_metadynamics

#TODO: Temporary aliases, to be deleted
OpenMM_box_relaxation = OpenMM_box_equilibration
small_molecule_parameterizor=small_molecule_parameterizer

from .modules.module_MM import NonBondedTheory, UFFdict, UFF_modH_dict, LJCoulpy, coulombcharge, LennardJones, \
    LJCoulombv2, LJCoulomb, MMforcefield_read
#MDtraj
from .interfaces.interface_mdtraj import MDtraj_imagetraj, MDtraj_slice, MDtraj_RMSF, MDtraj_coord_analyze

# QM/MM
from .modules.module_QMMM import QMMMTheory, actregiondefine, read_charges_from_psf
from .modules.module_polembed import PolEmbedTheory

# Knarr
from .interfaces.interface_knarr import NEB, NEBTS

#VMD
from .interfaces.interface_VMD import write_VMD_script_cube

# ASE-Dynamics
from .interfaces.interface_ASE import Dynamics_ASE

# Plumed interface
from .interfaces.interface_plumed import plumed_ASH, MTD_analyze

# Solvation
# NOTE: module_solvation.py or module_solvation2.py To be cleaned up
import ash.functions.functions_solv

# Molcrys
import ash.modules.module_molcrys
from .modules.module_molcrys import molcrys, Fragmenttype

# Geometry optimization
from .functions.functions_optimization import SimpleOpt, BernyOpt

#geomeTRIC interface
from .interfaces.interface_geometric_new import geomeTRICOptimizer,GeomeTRICOptimizerClass
Optimizer = geomeTRICOptimizer
Opt = geomeTRICOptimizer

# PES
from .modules.module_PES_rewrite import PhotoElectron, potential_adjustor_DFT, plot_PES_Spectrum,Read_old_PES_results

# Workflows, benchmarking etc
import ash.modules.module_workflows
import ash.modules.module_highlevel_workflows
from .modules.module_highlevel_workflows import ORCA_CC_CBS_Theory, Reaction_FCI_Analysis, make_ICE_theory, Auto_ICE_CAS,ORCA_MRCI_CBS_Theory

CC_CBS_Theory = ORCA_CC_CBS_Theory #TODO: Temporary alias

from .modules.module_workflows import ReactionEnergy, thermochemprotocol_reaction, thermochemprotocol_single, \
    confsampler_protocol, auto_active_space, calc_xyzfiles, ProjectResults, Reaction_Highlevel_Analysis, FormationEnthalpy, \
    AutoNonAufbau, ExcitedStateSCFOptimizer,TDDFT_vib_ave
import ash.modules.module_benchmarking
from .modules.module_benchmarking import run_benchmark


#Plotting
import ash.modules.module_plotting
from .modules.module_plotting import reactionprofile_plot, contourplot, plot_Spectrum, MOplot_vertical, ASH_plot

# Other
import ash.interfaces.interface_crest
from .interfaces.interface_crest import call_crest, call_crest_entropy, get_crest_conformers

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
#As PythonCall can be a bit slow to load, it is best to only load when needed (current behaviour)
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
