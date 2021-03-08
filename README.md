
<img src="ash-simple-logo-letterbig.png" alt="drawing" width="300" align="right"/>

 # Ash: a computational chemistry environment
Ash is a Python-based computational chemistry and QM/MM environment, primarily for molecular calculations
in the gas phase, explicit solution, crystal or protein environment. Can do single-point calculations,
geometry optimizations, molecular dynamics (soon), numerical frequencies using a MM, QM or QM/MM Hamiltonian.
Interfaces to popular QM codes: ORCA, xTB, Psi4, PySCF

Documentation: https://ash.readthedocs.io/en/latest



Example:
```sh
from ash import *
import sys
settings_ash.init()

coords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
#Create fragment
HF_frag=Fragment(coordsstring=coords)

#ORCA settings
orcadir='/opt/orca_4.2.1'
input="! BP86 def2-SVP Grid5 Finalgrid6 tightscf"
blocks="%scf maxiter 200 end"
#ORCA theory object
ORCAcalc = ORCATheory(orcadir=orcadir, fragment=HF_frag, charge=0, mult=1,
                         	orcasimpleinput=input, orcablocks=blocks)
#Call optimizer with ORCAtheory object and fragment
geomeTRICOptimizer(ORCAcalc,HF_frag)
 ```
