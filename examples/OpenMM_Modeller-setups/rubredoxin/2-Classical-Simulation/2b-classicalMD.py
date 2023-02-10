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

#MM minimization for 100 steps
OpenMM_Opt(fragment=fragment, theory=omm, maxiter=100, tolerance=1)

#NPT simulation until density and volume converges
OpenMM_box_relaxation(fragment=fragment, theory=omm, datafilename="nptsim.csv", numsteps_per_NPT=10000,
                        volume_threshold=1.0, density_threshold=0.001, temperature=300, timestep=0.004,
                        traj_frequency=100, trajfilename='relaxbox_NPT', trajectory_file_option='DCD', coupling_frequency=1)

#NVT MD simulation for 1 ns
OpenMM_MD(fragment=fragment, theory=omm, timestep=0.004, simulation_time=1000, traj_frequency=1000, temperature=300,
    integrator='LangevinMiddleIntegrator', coupling_frequency=1, trajectory_file_option='DCD')

#Note. This script will run a classical MD simulation for 1 ns. You may want to submit to the cluster instead.
#NOTE: Change platofrm to 'OpenCL' or 'CUDA' if GPU cores are available (much faster).
