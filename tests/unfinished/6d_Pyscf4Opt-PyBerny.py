from ash import *
import sys
#Define global system settings ( scale, tol and conndepth keywords for connectivity)

#Add coordinates to fragment
H2O_frag=Fragment(xyzfile='/Users/bjornsson/ownCloud/ASH-tests/testsuite/h2o_strained.xyz')

PySCFTheorycalculation = PySCFTheory(charge=0, mult=1,
    pyscfbasis='def2-SVP', pyscffunctional='bp86', printsetting=False)


#Using PyBernyOpt optimizer from Github
BernyOpt(PySCFTheorycalculation,H2O_frag)

print("Optimized coordinates:")
H2O_frag.print_coords()


sys.exit(0)
