from ash import *

#########################################################################
# Protocol to setup a new solute-counterion-solvent system from scratch
#########################################################################

numcores=1

# Solvent input parameters
box_size=50
padpar=2.0 # padding parameter (Å) for PBC. Increase if simulation crashes
solvent_density = 1.16  # g/ml
solvent_xyzfile="dfb.xyz"
solvent_name="DFB" #Name used for files
solvent_FF="OpenFF"
max_NPT_cycles=10
# Solute input parameters
solute_xyzfile="Rh+_cctbu_pnp_opt_reoriented.xyz"
solute_name="TMC"
solute_charge=1
solute_mult=2
solute_dft_string="r2SCAN-3c"
#Counterion input parameters
counterion_xyzfile="BPHCF3_4_counter.xyz"
counterion_name="ION"
counterion_charge=-1
counterion_mult=1
counterion_dft_string="r2SCAN-3c"
############################################

###############################
# SOLVENT FF
###############################
# Parameterize small molecule using OpenFF
small_molecule_parameterizer(forcefield_option=solvent_FF, xyzfile=solvent_xyzfile,
            charge=0,resname=solvent_name)


###############################
# SOLUTE FF
###############################
# Create a nonbonded FF for molecule
solute = Fragment(xyzfile=solute_xyzfile, charge=solute_charge, mult=solute_mult)

# Defining QM-theory to be used for charge calculation
qm = ORCATheory(orcasimpleinput=f"! {solute_dft_string} tightscf", numcores=numcores)
#
write_nonbonded_FF_for_ligand(fragment=solute, resname=solute_name, theory=qm,
        coulomb14scale=1.0, lj14scale=1.0, charge_model="CM5_ORCA")

# Write PDB-file (TMC.pdb) for later use, skipping connectivity lines
solute.write_pdbfile_openmm("TMC", skip_connectivity=True, resname=solute_name)
qm.cleanup()
# Create a nonbonded FF for counterion
counterion = Fragment(xyzfile=counterion_xyzfile, charge=counterion_charge, mult=counterion_mult)
qm = ORCATheory(orcasimpleinput=f"! {counterion_dft_string} tightscf", numcores=numcores)
write_nonbonded_FF_for_ligand(fragment=counterion, resname="ION", theory=qm,
        coulomb14scale=1.0, lj14scale=1.0, charge_model="CM5_ORCA")
# Write PDB-file for later use, skipping connectivity lines
counterion.write_pdbfile_openmm("ION", skip_connectivity=True, resname="ION")

###############################
# SOLVENT BOX
###############################
# Create a X Å cubic box of acetonitrile molecules corresponding to a density of 0.786 g/ml
packmol_solvate(inputfiles=[f"{solvent_name}.pdb"], density=solvent_density,
    min_coordinates=[0,0,0], max_coordinates=[box_size,box_size,box_size])

# NPT equilibration. Will give optimal box dimensions
pdbfile="final_withcon.pdb"
solventbox = Fragment(pdbfile=pdbfile)
# Note: using slightly larger box dimensions (55 instead of 50) to avoid initial periodicity problems at boundary
omm = OpenMMTheory(xmlfiles=[f"openff_{solvent_name}.xml"],pdbfile=pdbfile, platform="OpenCL",
            periodic=True, autoconstraints='HBonds', periodic_cell_dimensions=[box_size+padpar,box_size+padpar,box_size+padpar,90.0,90.0,90.0])
# NPT equilibration. Note: platform='OpenCL' (or CUDA if NVIDIA GPU) runs OpenMM on GPU, should run quite fast even on laptop.
# Use platform='CPU' if no GPU available
OpenMM_box_equilibration(fragment=solventbox, theory=omm, datafilename="nptsim.csv", timestep=0.001,
                                numsteps_per_NPT=10000,max_NPT_cycles=max_NPT_cycles,traj_frequency=100,
                                volume_threshold=1.0, density_threshold=0.01, temperature=300)

###############################
# INSERTION
###############################
solvent_pdbfile="equilibration_NPT_lastframe.pdb" # using NPT-equilibrated solvent-box

# Inserting solute into solvent-box and get new solution fragment and file solution.pdb
solution = insert_solute_into_solvent(solvent_pdb=solvent_pdbfile,
            solute_pdb=f"{solute_name}.pdb", solute2_pdb=f"{counterion_name}.pdb",
            write_pdb=True)

# Final Test to see if OpenMMTheory object can be defined from XML-files and final system file (solution.pdb)
omm = OpenMMTheory(xmlfiles=[f"{solute_name}.xml",f"{counterion_name}.xml", f"openff_{solvent_name}.xml"],pdbfile="solution.pdb", periodic=True,
    autoconstraints=None, rigidwater=False)

# Now we are ready to do QM/MM
#Defining QM/MM object to check
qm_theory = xTBTheory(xtbmethod="GFN2")
qmatoms = list(range(0,solute.numatoms+counterion.numatoms))
qm_mm = QMMMTheory(qm_theory=qm_theory, mm_theory=omm, qmatoms=qmatoms, fragment=solution,
    qm_charge=0, qm_mult=2)

# And the we could run something: e.g.
# Singlepoint(fragment=solution, theory=qm_mm)
#Optimizer(fragment=solution, theory=qm_mm, ActiveRegion=True, actatoms=qmatoms)
MolecularDynamics(fragment=solution, theory=qm_mm, timestep=0.001, simulation_time=10, traj_frequency=1,
    temperature=300, integrator='LangevinMiddleIntegrator', coupling_frequency=1,
    trajfilename='QM_MM_NVT-MD',trajectory_file_option='DCD')