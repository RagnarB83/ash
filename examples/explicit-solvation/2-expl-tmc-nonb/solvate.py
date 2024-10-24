from ash import *

mol = Fragment(xyzfile="fecl4.xyz", charge=-1, mult=6)
#Solvate molecule in a 30x30x30 Å TIP3P water box.
solvate_small_molecule(fragment=mol, xmlfile="LIG.xml", watermodel='tip3p', solvent_boxdims=[30,30,30])
