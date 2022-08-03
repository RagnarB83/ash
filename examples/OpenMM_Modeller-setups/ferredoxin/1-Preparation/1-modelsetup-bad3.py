from ash import *

#Original raw PDB-file (no hydrogens, nosolvent)
pdbfile="6lk1.pdb"

#XML-file to deal with FeS cluster
extraxmlfile="specialresidue.xml"

# Setting up system via Modeller
OpenMM_Modeller(pdbfile=pdbfile,forcefield="CHARMM36", use_higher_occupancy=True,
    extraxmlfile="specialresidue.xml")


