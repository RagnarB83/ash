from ash import *

#Define number of cores variable
numcores=3

#Fe(SCH2)4 indices (inspect system_aftersolvent.pdb file to get indices)
qmatoms=[93,94,95,96,133,134,135,136,564,565,566,567,604,605,606,607,755]

#Defining fragment containing coordinates (can be read from XYZ-file, ASH fragment, PDB-file)
lastpdbfile="final_MDfrag_laststep_imaged.pdb"
fragment=Fragment(pdbfile=lastpdbfile)

#Creating new OpenMM object from OpenMM XML files (built-in CHARMM36 and a user-defined one)
omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", "./specialresidue.xml"], pdbfile="finalsystem.pdb", periodic=True,
            platform='OpenCL', numcores=numcores, autoconstraints='HBonds',  rigidwater=True)

#QM theory
xtbobject = xTBTheory(charge=-1, mult=6, xtbmethod="GFN1", numcores=numcores)
#QM/MM theory
qmmm = QMMMTheory(qm_theory=xtbobject, mm_theory=omm, fragment=fragment,
        embedding="Elstat", qmatoms=qmatoms, printlevel=1)

#QM/MM MD simulation for 5 ps
OpenMM_MD(fragment=fragment, theory=qmmm, timestep=0.001, simulation_time=5, traj_frequency=50, temperature=300,
    integrator='LangevinMiddleIntegrator', coupling_frequency=1, trajectory_file_option='DCD')

#Re-image trajectory so that protein is in middle
MDtraj_imagetraj("trajectory.dcd", "final_MDfrag_laststep.pdb", format='DCD')