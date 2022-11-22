from ash import *
import sys
#Define global system settings ( scale, tol and conndepth keywords for connectivity)

fragcoords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""

#Add coordinates to fragment
HF_frag=Fragment(coordsstring=fragcoords)

PySCFTheorycalculation = PySCFTheory(fragment=HF_frag, charge=0, mult=1,
    pyscfbasis='def2-SVP', pyscffunctional='bp86', printsetting=False)

#Using geomeTRIC optimization from Github
Optimizer(PySCFTheorycalculation,HF_frag, coordsystem='tric')



sys.exit(0)
