from ash import *

#Original raw PDB-file (no hydrogens, nosolvent)
pdbfile="2dsx.pdb"

#XML-file to deal with cofactor
extraxmlfile="./specialresidue.xml"

#Setting some manual protonation states.
#Here defining residues in chain A with resid values: 6,9,39,42 to be deprotonated cysteines (CYX). 
#NOTE: Here the actual resid values in the PDB-file are used (ASH's 0-based indexing does not apply)
residue_variants={'A':{6:'CYX',9:'CYX',39:'CYX',42:'CYX'}}

# Setting up system via Modeller
OpenMM_Modeller(pdbfile=pdbfile, forcefield='CHARMM36',
    extraxmlfile=extraxmlfile, watermodel="tip3p", pH=7.0, solvent_padding=10.0,
    ionicstrength=0.1, pos_iontype='Na+', neg_iontype='Cl-', residue_variants=residue_variants)
