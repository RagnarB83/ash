from ash import *
import sys
#Define global system settings ( scale, tol and conndepth keywords for connectivity)

#Psi4 Library interface. Requires importing psi4 library into Ash
#Using printoption for writing to file with no stdout to screen
#Using Psi4 OpenMP parallelization

fragcoords="""
C       -0.194584000     -0.478174000     -1.973643000
C       -1.542366000     -0.474065000     -1.635446000
C        0.724198000      0.229848000     -1.216614000
H       -2.255565000     -1.031130000     -2.231587000
H        1.772188000      0.225066000     -1.482255000
C       -1.969238000      0.246736000     -0.530605000
C        0.280852000      0.948265000     -0.111396000
H        0.987230000      1.507866000      0.487608000
C       -1.058837000      0.962816000      0.237788000
H       -1.415784000      1.520844000      1.091298000
H        0.131968000     -1.042144000     -2.837420000
O       -3.274587000      0.282263000     -0.153446000
H       -3.811149000     -0.196258000     -0.767527000
"""

#Add coordinates to fragment
PhenolH2O_frag=Fragment(coordsstring=fragcoords)

#Defining DFT functional passed to Psi4 interface. Used in Psi4 energy('scf', density_psi4method=X) command.
psi4method='wb97X'

#Psi4 dictionary with basic SCF options
psi4dictvar={
'reference': 'uhf',
'basis' : 'def2-SVP',
'pe' : 'true',
'scf_type' : 'pk'}


Psi4SPcalculation = Psi4Theory(fragment=PhenolH2O_frag, charge=0, mult=1, psi4settings=psi4dictvar, numcores=4,
    psi4method=psi4method, runmode='library', pe=True, potfile='phenolacceptor-FULL-H2O-n-MOD.pot', printsetting=True)

#Simple Energy SP calc.
Singlepoint(fragment=PhenolH2O_frag, theory=Psi4SPcalculation)
print("")
print(Psi4SPcalculation.__dict__)
print(Psi4SPcalculation.energy)


#Clean 
Psi4SPcalculation.cleanup()

sys.exit(0)
