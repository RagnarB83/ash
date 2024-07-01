from ash import *

#Original raw PDB-file (no hydrogens, nosolvent)
#Download from https://www.rcsb.org/structure/1AKI
pdbfile="1aki.pdb"

#Setting up new system, adding hydrogens, solvent, ions and defining forcefield, topology
#Here using CHARMM, Amber is also possible
omm, ashfrag = OpenMM_Modeller(pdbfile=pdbfile, forcefield='CHARMM36', watermodel="tip3p", pH=7.0,
    solvent_padding=10.0, ionicstrength=0.1, platform='OpenCL')

#Gentle warmup MD (3 MD simulations: 10/50/200 steps with timesteps 0.5/1/4 fs at 1 K/10K/300K)
Gentle_warm_up_MD(fragment=ashfrag, theory=omm, time_steps=[0.0005,0.001,0.004],
            steps=[10,50,200], temperatures=[1,10,300])

#Run NPT simulation until density and volume converges
OpenMM_box_equilibration(fragment=ashfrag, theory=omm, datafilename="nptsim.csv", numsteps_per_NPT=10000,
                  temperature=300, timestep=0.003, traj_frequency=100, trajfilename='equilbox_NPT',
                  trajectory_file_option='DCD', coupling_frequency=1)
