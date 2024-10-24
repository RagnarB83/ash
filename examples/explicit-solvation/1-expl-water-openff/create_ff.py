from ash import *

#Parameterize small molecule using OpenFF
small_molecule_parameterizer(xyzfile="isobutyraldehyde.xyz", forcefield_option="OpenFF", charge=0)
