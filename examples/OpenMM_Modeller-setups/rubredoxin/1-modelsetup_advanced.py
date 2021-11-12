from ash import *

#Original raw PDB-file (no hydrogens, nosolvent)
pdbfile="2dsx.pdb"

#XML-file to deal with cofactor
extraxmlfile="./specialresidue.xml"

#Setting some manual protonation states. Note ASH, counts from 0
#Here defining residues 5,8,38,41 (6,9,39,42 in the PDB-file) to be deprotonated cysteines (CYX).
residue_variants={5:'CYX',8:'CYX',38:'CYX',41:'CYX'}

# Setting up system via Modeller
OpenMM_Modeller(pdbfile=pdbfile, forcefield='CHARMM36',
    extraxmlfile=extraxmlfile, watermodel="tip3p", pH=7.0, solvent_padding=10.0,
    ionicstrength=0.1, pos_iontype='Na+', neg_iontype='Cl-', residue_variants=residue_variants)
