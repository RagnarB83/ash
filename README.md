
<img src="ash-simple-logo-letterbig.png" alt="drawing" width="300" align="right"/>

 # ASH: a multiscale modelling program
ASH is a Python-based computational chemistry and QM/MM environment for molecular calculations in the gas phase, explicit solution, crystal or protein environment. Can do single-point calculations, geometry optimizations, nudged elastic band calculations, surface scans, molecular dynamics, numerical frequencies and many other things using a MM, QM or QM/MM Hamiltonian.
Interfaces to popular QM codes: ORCA, xTB, PySCF, MRCC, Psi4, Dalton, CFour, TeraChem, QUICK. Interface to the OpenMM library for MM and MD algorithms. Interfaces to specialized codes like Block, Dice and ipie for DMRG, SHCI and AFQMC calculations.
Excellent environment for writing simple or complex computational chemistry workflows.

While ASH is ready to be used in computational chemistry research, it is a young project and there will probably be some issues and bugs to be discovered if you start using it.

**In case of problems:** 
Please open an issue on Github and I will try to fix any problems as soon as possible.


**Documentation:**

 https://ash.readthedocs.io


**Development:**

ASH welcomes any contributions.

Ongoing priorities:
- Improve documentation of code, write docstrings.
- Write unit tests
- Rewrite silly old code.
- Reduce code redundancy.


**Example:**

```sh
from ash import *

coords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
#Create fragment from multi-line string
HF_frag=Fragment(coordsstring=coords, charge=0, mult=1)
#Alternative: Create fragment from XYZ-file
HF_frag2=Fragment(xyzfile="hf.xyz", charge=0, mult=1)

#Define ORCA theory settings strings
input="! r2SCAN def2-SVP def2/J tightscf"
blocks="%scf maxiter 200 end"
#Define ORCA theory object
ORCAcalc = ORCATheory(orcasimpleinput=input, orcablocks=blocks)

#Call optimizer with ORCAtheory object and fragment as input
Optimizer(theory=ORCAcalc,fragment=HF_frag)

#Numerical frequencies
NumFreq(theory=ORCAcalc,fragment=HF_frag)

#Singlepoint calculation
Singlepoint(theory=ORCAcalc,fragment=HF_frag)

 ```
