from ash import *
import sys
#Define global system settings ( scale, tol and conndepth keywords for connectivity)

#Psi4 Library interface. Requires importing psi4 library into Ash
#Using printoption for writing to file with no stdout to screen
#Using Psi4 OpenMP parallelization

fragcoords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""

#Add coordinates to fragment
HF_frag=Fragment(coordsstring=fragcoords)

#Defining DFT functional passed to Psi4 interface. Used in Psi4 energy('scf', density_psi4method=X) command.
psi4method='bp86'

#Psi4 dictionary with basic SCF options
psi4dictvar={
'reference': 'uhf',
'basis' : 'def2-SVP',
'scf_type' : 'pk'}


Psi4SPcalculation = Psi4Theory(fragment=HF_frag, charge=0, mult=1, psi4settings=psi4dictvar, psi4method=psi4method, runmode='psithon', printsetting=True, numcores=4)

#Simple Energy SP calc
Singlepoint(theory=Psi4SPcalculation,fragment=HF_frag)
print(Psi4SPcalculation.__dict__)
print(Psi4SPcalculation.energy)

#Energy+Gradient calculation
Singlepoint(theory=Psi4SPcalculation,fragment=HF_frag,Grad=True)
print(Psi4SPcalculation.__dict__)
print(Psi4SPcalculation.energy)
print(Psi4SPcalculation.gradient)

#Clean 
Psi4SPcalculation.cleanup()

sys.exit(0)
