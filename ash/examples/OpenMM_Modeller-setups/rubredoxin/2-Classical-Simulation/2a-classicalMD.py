from ash import *

numcores=4

#FeS4 indices (inspect system_aftersolvent.pdb file to get indices)
cofactor_indices=[96, 136, 567, 607, 755]
bondconstraints=[[755,96],[755,136],[755,567],[755,607]]

#Defining fragment containing coordinates (can be read from XYZ-file, ASH fragment, PDB-file)
pdbfile="finalsystem.pdb"
fragment=Fragment(pdbfile=pdbfile)

#Creating new OpenMM object from OpenMM XML files (built-in CHARMM36 and a user-defined one)
omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", "./specialresidue.xml"], pdbfile=pdbfile, periodic=True,
            platform='CPU', numcores=numcores, autoconstraints='HBonds', constraints=bondconstraints, rigidwater=True)

#MM minimization to get rid the worst contacts
OpenMM_Opt(fragment=fragment, theory=omm, maxiter=100, tolerance=1000)

#Classical MD simulation for 5 ps
OpenMM_MD(fragment=fragment, theory=omm, timestep=0.001, simulation_time=5, traj_frequency=10, temperature=300,
    integrator='LangevinMiddleIntegrator', coupling_frequency=1, trajectory_file_option='DCD')

#Re-image trajectory so that protein is in middle
MDtraj_imagetraj("trajectory.dcd", "final_MDfrag_laststep.pdb", format='DCD')

#Note. This script will run a classical MD simulation for 5 ps only
