from ash import *

#Original raw PDB-file (no hydrogens, nosolvent)
pdbfile="6lk1-mod.pdb"

#XML-file to deal with FeS cluster
extraxmlfile="specialresidue.xml"

#Defining deptonated cysteine residues
residue_variants={'A':{38:'CYX',43:'CYX',46:'CYX',76:'CYX'},'B':{38:'CYX',43:'CYX',46:'CYX',76:'CYX'}}

# Setting up system via Modeller
OpenMM_Modeller(pdbfile=pdbfile,forcefield="CHARMM36", use_higher_occupancy=True,
    extraxmlfile="specialresidue.xml",  residue_variants=residue_variants)

