from ash import *
import sys
#Define global system settings ( scale, tol and conndepth keywords for connectivity)


#Defining DFT functional passed to Psi4 interface. Used in Psi4 energy('scf', density_psi4method=X) command.
psi4method='bp86'

#Psi4 dictionary with basic SCF options
psi4dictvar={
'reference' : 'rhf',
'basis' : 'def2-SVP',
'scf_type' : 'pk'}

fragcoords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""

#Add coordinates to fragment
HF_frag=Fragment(coordsstring=fragcoords)

Psi4SPcalculation = Psi4Theory(fragment=HF_frag, charge=0, mult=1, psi4settings=psi4dictvar, psi4method=psi4method, runmode='library')


#Using geomeTRIC optimization from Github
Optimizer(Psi4SPcalculation,HF_frag)



sys.exit(0)
