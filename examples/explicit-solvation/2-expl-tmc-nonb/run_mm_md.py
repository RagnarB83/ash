from ash import *

#Read in coordinates of the full system
pdbfile="system_aftersolvent.pdb"
fragment = Fragment(pdbfile=pdbfile)

#Constrain the bonds of the ligand.
#Note: additional angle and dihedral constraints may also be appropriate
bondconstraints = [[0,1],[0,2],[0,3],[0,4]]

#Create an OpenMMTheory object based on PDB-file and XML-files for water and small-molecule
omm =OpenMMTheory(xmlfiles=["LIG.xml", "amber/tip3p_standard.xml"],
            pdbfile=pdbfile, periodic=True, rigidwater=True, autoconstraints='HBonds',
            constraints=bondconstraints)

#Gently warms up the system
Gentle_warm_up_MD(fragment=fragment, theory=omm, use_mdtraj=False)

#Equilibrates the system via a multi-step NPT simulation.
#This changes the box size of the system until volume and density have converged.
#Note that thresholds for volume and density may have to be adjusted
OpenMM_box_equilibration(fragment=fragment, theory=omm, datafilename="nptsim.csv", timestep=0.004,
                            numsteps_per_NPT=10000,max_NPT_cycles=10,traj_frequency=100,
                            volume_threshold=1.0, density_threshold=0.01, temperature=300)
