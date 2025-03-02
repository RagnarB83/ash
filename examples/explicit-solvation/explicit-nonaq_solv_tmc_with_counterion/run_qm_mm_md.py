from ash import *

#Script to run QM/MM MD after system setup (see create_system.py)

numcores=1
###############################
solvent_name="DFB"
solution = Fragment(pdbfile="solution.pdb")
solute_xmlfile="TMC.xml"
solvent_xmlfile=f"openff_{solvent_name}.xml"
counterion_xmlfile="ION.xml"
solute_numatoms=154 #Defining TMC + counterion
charge=0 #Total QM-region charge
mult=2 #QM-region multiplicity
openmm_platform="OpenCL" # Platform to run OpenMM on: 'CPU', 'OpenCL' or 'CUDA

# Now we are ready to do QM/MM
qm_theory = xTBTheory(xtbmethod="GFN2")
qmatoms = list(range(0,solute_numatoms))

# OpenMM object for MM sim (frozen QM-region)
omm_for_mm = OpenMMTheory(xmlfiles=[solute_xmlfile,counterion_xmlfile, solvent_xmlfile],pdbfile="solution.pdb", periodic=True,
    autoconstraints='HBonds', rigidwater=True, platform=openmm_platform, frozen_atoms=qmatoms)

# OpenMM objectfor QM/MM sim
omm = OpenMMTheory(xmlfiles=[solute_xmlfile, counterion_xmlfile, solvent_xmlfile],pdbfile="solution.pdb", periodic=True,
    autoconstraints=None, rigidwater=False, platform=openmm_platform)

# QM/MM
qm_mm = QMMMTheory(qm_theory=qm_theory, mm_theory=omm, qmatoms=qmatoms, fragment=solution,
    qm_charge=charge, qm_mult=mult)

# Run a gentle classical warmup MD with qmatoms frozen
Gentle_warm_up_MD(theory=omm_for_mm, fragment=solution, time_steps=[0.0005,0.001,0.001],
                  steps=[10,50,1000], temperatures=[1,5,10])

#Run the QM/MM MD
MolecularDynamics(fragment=solution, theory=qm_mm, timestep=0.001, simulation_time=10, traj_frequency=100,
    temperature=300, integrator='LangevinMiddleIntegrator', coupling_frequency=1,
    trajfilename='QM_MM_NVT-MD',trajectory_file_option='DCD')