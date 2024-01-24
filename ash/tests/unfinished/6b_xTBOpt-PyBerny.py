from ash import *
import sys

#Define global system settings ( scale, tol and conndepth keywords for connectivity)

#xTB
xtbdir="/opt/xtb-6.4.0/bin"
xtbmethod='GFN2'

#Add coordinates to fragment
H2O_frag=Fragment(xyzfile='/Users/bjornsson/ownCloud/ASH-tests/testsuite/h2o_strained.xyz')

xtbcalc = xTBTheory(xtbdir, fragment=H2O_frag, charge=0, mult=1, xtbmethod=xtbmethod)

#Using PyBernyOpt optimizer from Github
BernyOpt(xtbcalc,H2O_frag)

print("Optimized coordinates:")
H2O_frag.print_coords()

sys.exit(0)
