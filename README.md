
<img src="ash-simple-logo-letterbig.png" alt="drawing" width="300" align="right"/>

 # ASH: a computational chemistry environment
ASH is a Python-based computational chemistry and QM/MM environment, primarily for molecular calculations
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
#Create fragment from multi-line string
HF_frag=Fragment(coordsstring=coords, charge=0, mult=1)

#Alternative: Create fragment from XYZ-file
HF_frag=Fragment(xyzfile="hf.xyz", charge=0, mult=1)

#Define ORCA theory settings strings
input="! r2SCAN def2-SVP def2/J tightscf"
blocks="%scf maxiter 200 end"
#Define ORCA theory object
ORCAcalc = ORCATheory(orcasimpleinput=input, orcablocks=blocks)

#Call optimizer with ORCAtheory object and fragment as input
geomeTRICOptimizer(theory=ORCAcalc,fragment=HF_frag)



 ```
