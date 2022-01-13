from ash import *

#Define pdbfile that points to the original raw PDB-file (typically without hydrogens and solvent)
pdbfile="2dsx.pdb"

# Setting up system via OpenMM_Modeller and requesting the CHARMM36 forcefield
OpenMM_Modeller(pdbfile=pdbfile, forcefield='CHARMM36', use_higher_occupancy=True)

#NOTE: This script will fail for 2dsx.pdb due to OpenMM not the recognizing FE residue.
