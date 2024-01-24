from ash import *

#Original raw PDB-file (no hydrogens, nosolvent)
pdbfile="6lk1.pdb"

# Setting up system via Modeller
OpenMM_Modeller(pdbfile=pdbfile,forcefield="CHARMM36")


