from ash import *

#Cores to use for OpenMM and QM/MM
numcores=4

#Original raw PDB-file (no hydrogens, nosolvent). Lysozyme example
pdbfile="1aki.pdb"


#Defining residues with special user-wanted protonation states
residue_variants={}

#Setting up new system, adding hydrogens, solvent, ions and defining forcefield, topology
openmmobject, ashfragment = OpenMM_Modeller(pdbfile=pdbfile, forcefield='CHARMM36', watermodel="tip3p", pH=7.0, 
    solvent_padding=10.0, ionicstrength=0.1, residue_variants=residue_variants)

#Alternatively: openmmobject can be recreated like this:
#openmmobject = OpenMMTheory(xmlfiles=[charmm36.xml, charmm36/water.xml], pdbfile="finalsystem.pdb", periodic=True)

#MM minimization for 100 steps
OpenMM_Opt(fragment=ashfragment, theory=openmmobject, maxiter=100, tolerance=1)

#Gentle heating-up protocol
Gentle_warm_up_MD(theory=openmmobject, fragment=ashfragment, 
    time_steps=[0.0005,0.001,0.004], steps=[10,50,10000], temperatures=[1,10,300])

#Classical MD simulation for 10 ps
OpenMM_MD(fragment=ashfragment, theory=openmmobject, timestep=0.001, simulation_time=10, traj_frequency=100, temperature=300,
    integrator='LangevinMiddleIntegrator', coupling_frequency=1, trajectory_file_option='DCD')

#Setting up QM/MM model
#QM-region: side-chain of ASP66
qmatomlist = [1013,1014,1015,1016,1017,1018]

#Define QM-theory. Here ORCA and r2SCAN-3c
ORCAinpline="! r2SCAN-3c tightscf"
ORCAblocklines="""
%maxcore 2000
%scf
MaxIter 500
end
"""
orcaobject = ORCATheory(orcasimpleinput=ORCAinpline,
                        orcablocks=ORCAblocklines, numcores=1)

# Create QM/MM OBJECT
qmmmobject = QMMMTheory(qm_theory=orcaobject, mm_theory=openmmobject,
    fragment=ashfragment, embedding="Elstat", qmatoms=qmatomlist, printlevel=2)

# QM/MM geometry optimization
Optimizer(theory=qmmmobject, fragment=ashfragment, ActiveRegion=True, actatoms=qmatomlist, maxiter=500,
    charge=-1,mult=1)
