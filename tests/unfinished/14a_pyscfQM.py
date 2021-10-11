from ash import *
import sys
#Define global system settings ( scale, tol and conndepth keywords for connectivity)

#PySCF Library interface. Requires importing pyscf library into Ash

fragcoords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
H 3.0 3.0 3.0
F 3.0 3.0 4.0
"""

#Add coordinates to fragment
HF_frag=Fragment(coordsstring=fragcoords)

#Defining DFT functional passed to PySCF interface.
pyscffunctional='b3lyp'

#PySCF basis. Can be string or dict object (elem-specific info)
pyscfbasis='def2-QZVP'

PySCFTheorycalculation = PySCFTheory(fragment=HF_frag, charge=0, mult=1, 
    pyscfbasis=pyscfbasis, pyscffunctional=pyscffunctional, printsetting=False)

Singlepoint(theory=PySCFTheorycalculation, fragment=HF_frag)


#Clean 
PySCFTheorycalculation.cleanup()

sys.exit(0)
