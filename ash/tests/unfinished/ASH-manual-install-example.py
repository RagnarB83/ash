from ash import *

#Create H2O fragment
coords="""
O       -1.377626260      0.000000000     -1.740199718
H       -1.377626260      0.759337000     -1.144156718
H       -1.377626260     -0.759337000     -1.144156718
"""
H2Ofragment=Fragment(coordsstring=coords)
#Defining ORCA-related variables
orcasimpleinput="! r2SCAN-3c tightscf"
orcablocks="%scf maxiter 200 end"

ORCAcalc = ORCATheory(charge=0, mult=1,
                            orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

ORCAcalc.cleanup()

#Basic Cartesian optimization
Optimizer(fragment=H2Ofragment, theory=ORCAcalc, coordsystem='tric')
