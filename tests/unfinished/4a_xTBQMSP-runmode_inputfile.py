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

xTBSPcalculation = xTBTheory(xtbdir=xtbdir, fragment=HF_frag, charge=0, mult=1, xtbmethod=xtbmethod, runmode='inputfile')

#Simple Energy SP calc
Singlepoint(theory=xTBSPcalculation,fragment=HF_frag)
print(xTBSPcalculation)
print(xTBSPcalculation.energy)

#Energy+Gradient calculation
Singlepoint(theory=xTBSPcalculation,fragment=HF_frag,Grad=True)
print(xTBSPcalculation.energy)
print(xTBSPcalculation.grad)

sys.exit(0)
