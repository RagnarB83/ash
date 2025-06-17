**master:**
![example workflow](https://github.com/RagnarB83/ash/actions/workflows/python-app-conda.yml/badge.svg)
**NEW:**
![example branch parameter](https://github.com/RagnarB83/ash/actions/workflows/python-app-conda.yml/badge.svg?branch=NEW)

<img src="ash-simple-logo-letterbig.png" alt="drawing" width="300" align="right"/>

 # ASH: a multi-scale, multi-theory modelling program
ASH is a Python-based computational chemistry and QM/MM environment for molecular calculations in the gas phase, explicit solution, crystal or protein environment. It's a program for performing single-point calculations, geometry optimizations, nudged elastic band calculations, surface scans, molecular dynamics, numerical frequencies and many other things using a MM, QM, QM/MM or ONIOM Hamiltonian.
Interfaces to popular QM codes: ORCA, xTB, PySCF, MRCC, ccpy, Psi4, Dalton, CFour, TeraChem, QUICK. Interface to the OpenMM library for MM and MD algorithms. Interfaces to specialized high-level QM codes like Block, Dice and ipie for DMRG, SHCI and AFQMC calculations. Interfaces to machine-learning libraries like PyTorch and MLatom for using and training machine learning potentials.
Excellent environment for writing simple or complex computational chemistry workflows.

**In case of problems:**
Please open an issue on Github and we will try to fix any problems as soon as possible.

**Installation:**
See https://ash.readthedocs.io/en/latest/setup.html for detailed installation instructions.
A proper ASH installation should usually be done in a conda/mamba environment together with the OpenMM library.

Basic installation via pip:

```sh
#Install ASH using pip (default main branch)
pip install git+https://github.com/RagnarB83/ash.git

#Install the NEW (development) branch of ASH
pip install git+https://github.com/RagnarB83/ash.git@NEW
 ```


**Documentation:**

 https://ash.readthedocs.io


**Development:**

ASH welcomes any contributions.

Ongoing priorities:
- Improve packaging
- Prepare for 1.0 release
- Fix more Python faux pas
- Write unit tests
- Improve documentation of code, write docstrings.

**Basic example:**

```sh
from ash import *

coords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
#Create fragment from multi-line string
HF_frag=Fragment(coordsstring=coords, charge=0, mult=1)
#Alternative: Create fragment from XYZ-file
#HF_frag2=Fragment(xyzfile="hf.xyz", charge=0, mult=1)

#Create ORCATheory object
input="! r2SCAN def2-SVP def2/J tightscf"
blocks="%scf maxiter 200 end"
ORCAcalc = ORCATheory(orcasimpleinput=input, orcablocks=blocks)

#Singlepoint calculation
Singlepoint(theory=ORCAcalc,fragment=HF_frag)

#Call optimizer
Optimizer(theory=ORCAcalc,fragment=HF_frag)

#Numerical frequencies
NumFreq(theory=ORCAcalc,fragment=HF_frag)

#DFT Molecular dynamics simulation for 2 ps with a 0.001 ps (1 fs) timestep
MolecularDynamics(fragment=HF_frag, theory=ORCAcalc, timestep=0.001, simulation_time=2)

 ```

**QM/MM example:**

```sh
from ash import *

# Defining a fragment
fragment = Fragment(pdbfile="system.pdb")
# QM-method and QM-region
qm_orca = ORCATheory(orcasimpleinput="! r2SCAN-3c tightscf", numcores=8)
# MM Theory
omm  = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", "specialresidue.xml"], 
                    pdbfile="system.pdb", periodic=True)

# QM/MM Theory
qmatoms = [93,94,95,96,97,133,134,135, 2001,2002]
qm_mm = QMMMTheory(qm_theory= qm_orca, mm_theory= omm, fragment=fragment, 
                    qm_charge=-1, qm_mult=6,  qmatoms= qmatoms, printlevel=1)

# Geometry optimization
Optimizer(theory=qm_mm,fragment=fragment, actatoms=qmatoms)
# or Molecular dynamics
MolecularDynamics(fragment=fragment, theory=qm_mm, timestep=0.001, simulation_time=2)
 ```
