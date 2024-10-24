
from ash import *

numcores=1

# Defining solute and theory
mol = Fragment(xyzfile="3fgaba.xyz", charge=0, mult=1)
theory = xTBTheory(runmode='library', solvent="H2O")

########################################################################
# Calling solvate_small_molecule with fragment mol as input
# choosing TIP3P water solvent and box dimensions of 70x70x70 Angstrom
########################################################################

# Option 1: Using LIG.xml file (nonbonded FF parameters created by write_nonbonded_FF_for_ligand)
solvate_small_molecule(fragment=mol, xmlfile="LIG.xml", watermodel='tip3p', solvent_boxdims=[70,70,70])

# Option 2: Using openff_LIG.xml file (full FF via OpenFF, created by small_molecule_parameterizer)
# solvate_small_molecule(fragment=mol, xmlfile="openff_LIG.xml", watermodel='tip3p', solvent_boxdims=[70,70,70])