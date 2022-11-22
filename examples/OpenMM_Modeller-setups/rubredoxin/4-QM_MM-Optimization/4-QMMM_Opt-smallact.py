from ash import *

#Define number of cores variable
numcores=4

#Fe(SCH2)4 indices (inspect system_aftersolvent.pdb file to get indices)
qmatoms=[93,94,95,96,133,134,135,136,564,565,566,567,604,605,606,607,755]

#Defining fragment containing coordinates (can be read from XYZ-file, ASH fragment, PDB-file)
lastpdbfile="final_MDfrag_laststep_imaged.pdb"
fragment=Fragment(pdbfile=lastpdbfile)

#Creating new OpenMM object from OpenMM XML files (built-in CHARMM36 and a user-defined one)
omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", "./specialresidue.xml"], pdbfile=lastpdbfile, periodic=True,
            platform='CPU', numcores=numcores, autoconstraints=None, rigidwater=False)

#QM theory: r2SCAN-3c DFT-composite method using ORCA
orca = ORCATheory(orcasimpleinput="! r2SCAN-3c tightscf", numcores=1)
#QM/MM theory
qmmm = QMMMTheory(qm_theory=orca, mm_theory=omm, fragment=fragment,
        embedding="Elstat", qmatoms=qmatoms, printlevel=1)

# QM/MM geometry optimization
#Defining active region as QM-region
actatoms=qmatoms
Optimizer(fragment=fragment, theory=qmmm, ActiveRegion=True, actatoms=actatoms, maxiter=200,
        charge=-1, mult=6)
