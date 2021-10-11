from ash import *
import sys

#Define global system settings ( scale, tol and conndepth keywords for connectivity)

#xTB
xtbdir="/opt/xtb-6.4.0/bin"
xtbmethod='GFN2'


fragcoords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""

#Add coordinates to fragment
HF_frag=Fragment(coordsstring=fragcoords)

xtbcalc = xTBTheory(xtbdir, fragment=HF_frag, charge=0, mult=1, xtbmethod=xtbmethod)

#Basic Cartesian optimization with KNARR-LBFGS
#with Frozen atoms
SimpleOpt(fragment=HF_frag, theory=xtbcalc, optimizer='KNARR-LBFGS', frozen_atoms=[1])

print(HF_frag.energy)
print("Optimized coordinates:")
HF_frag.print_coords()

sys.exit(0)
