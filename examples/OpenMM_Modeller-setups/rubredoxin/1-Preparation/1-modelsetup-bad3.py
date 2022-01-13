from ash import *

#Define pdbfile that points to the original raw PDB-file (typically without hydrogens and solvent)
pdbfile="2dsx.pdb"

#XML-file to deal with cofactor
extraxmlfile="./specialresidue.xml"

# Setting up system via OpenMM_Modeller and requesting the CHARMM36 forcefield
OpenMM_Modeller(pdbfile=pdbfile, forcefield='CHARMM36', use_higher_occupancy=True, extraxmlfile=extraxmlfile)

#NOTE: This script will run to completion for 2dsx.pdb but will give wrong protonation states for the Fe-coordinates Cys residues
