
<img src="ash-simple-logo-letterbig.png" alt="drawing" width="300" align="right"/>

 # Ash: a computational chemistry environment
Ash is a Python-based computational chemistry and QM/MM environment, primarily for molecular calculations
in the gas phase, explicit solution, crystal or protein environment. Can do single-point calculations,
geometry optimizations, surface scans, molecular dynamics, numerical frequencies etc. using a MM, QM or QM/MM Hamiltonian.
Interfaces to popular QM codes: ORCA, xTB, Psi4, PySCF, Dalton, CFour, MRCC.

Documentation: https://ash.readthedocs.io/en/latest



Example:
```sh
from ash import *

coords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
#Create fragment from string
HF_frag=Fragment(coordsstring=coords)

#Alternative: Create fragment from XYZ-file
HF_frag=Fragment(xyzfile="hf.xyz")

#ORCA settings
orcadir='/opt/orca_5.0.0'
input="! r2SCAN def2-SVP def2/J tightscf"
blocks="%scf maxiter 200 end"
#ORCA theory object
ORCAcalc = ORCATheory(orcadir=orcadir, fragment=HF_frag, charge=0, mult=1,
                         	orcasimpleinput=input, orcablocks=blocks)
#Call optimizer with ORCAtheory object and fragment
geomeTRICOptimizer(ORCAcalc,HF_frag)



 ```
