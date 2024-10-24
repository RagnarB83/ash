from ash import *

mol = Fragment(xyzfile="isobutyraldehyde.xyz", charge=0, mult=1)
#Solvate molecule in a 30x30x30 Å TIP3P water box.
solvate_small_molecule(fragment=mol, xmlfile="openff_LIG.xml", watermodel='tip3p', solvent_boxdims=[30,30,30])
