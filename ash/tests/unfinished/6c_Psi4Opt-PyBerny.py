from ash import *
import sys
#Define global system settings ( scale, tol and conndepth keywords for connectivity)

#Add coordinates to fragment
H2O_frag=Fragment(xyzfile='/Users/bjornsson/ownCloud/ASH-tests/testsuite/h2o_strained.xyz')

#Defining DFT functional passed to Psi4 interface. Used in Psi4 energy('scf', density_psi4method=X) command.
psi4method='bp86'

#Psi4 dictionary with basic SCF options
psi4dictvar={
'reference' : 'rhf',
'basis' : 'def2-SVP',
'scf_type' : 'pk'}

Psi4SPcalculation = Psi4Theory(fragment=H2O_frag, charge=0, mult=1, psi4settings=psi4dictvar, psi4method=psi4method, runmode='library')


#Using PyBernyOpt optimizer from Github
BernyOpt(Psi4SPcalculation,H2O_frag)

print("Optimized coordinates:")
H2O_frag.print_coords()


sys.exit(0)
