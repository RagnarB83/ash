from ash import *

#Original raw PDB-file (no hydrogens, nosolvent)
pdbfile="2dsx.pdb"

#XML-file to deal with cofactor
extraxmlfile="./specialresidue.xml"

# Setting up system via Modeller
OpenMM_Modeller(pdbfile=pdbfile,forcefield="CHARMM36")
