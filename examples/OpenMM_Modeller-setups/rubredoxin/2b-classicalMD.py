from ash import *

numcores=4

#XML-file to deal with cofactor
extraxmlfile="./specialresidue.xml"

#FeS4 indices (inspect system_aftersolvent.pdb file to get indices)
cofactor_indices=[96, 136, 567, 607, 755]
bondconstraints=[[755,96],[755,136],[755,567],[755,607]]

#Defining fragment containing coordinates (can be read from XYZ-file, ASH fragment, PDB-file)
fragment=Fragment(pdbfile="finalsystem.pdb")

#Creating new OpenMM object from OpenMM full system file
omm = OpenMMTheory(xmlsystemfile="system_full.xml", pdbfile="finalsystem.pdb", periodic=True, platform='OpenCL', numcores=numcores,
                    autoconstraints='HBonds', constraints=bondconstraints, rigidwater=True)

#MM minimization for 100 steps
OpenMM_Opt(fragment=fragment, theory=omm, maxiter=100, tolerance=1)

#NPT simulation until density and volume converges
OpenMM_box_relaxation(fragment=fragment, theory=omm, datafilename="nptsim.csv", numsteps_per_NPT=10000,
                        volume_threshold=1.0, density_threshold=0.001, temperature=300, timestep=0.004,
                        traj_frequency=100, trajfilename='relaxbox_NPT', trajectory_file_option='DCD', coupling_frequency=1)

#NVT MD simulation for 1 ns
OpenMM_MD(fragment=fragment, theory=omm, timestep=0.004, simulation_time=1000, traj_frequency=1000, temperature=300,
    integrator='LangevinMiddleIntegrator', coupling_frequency=1, trajectory_file_option='DCD')
