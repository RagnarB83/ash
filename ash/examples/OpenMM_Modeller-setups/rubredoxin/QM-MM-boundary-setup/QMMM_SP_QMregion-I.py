from ash import *

#QM/MM single-point calculation to demonstrate different QM-regions
#
#QM-I: Fe and S atoms only: 5 QM-atoms + 4 linkatoms will be added

#Define number of cores variable
numcores=1

#QM-I:
qmatoms=[96,136,567,607,755]

#Defining fragment containing coordinates (can be read from XYZ-file, ASH fragment, PDB-file)
lastpdbfile="trajectory_lastframe.pdb"
fragment=Fragment(pdbfile=lastpdbfile)

#Creating new OpenMM object from OpenMM XML files (built-in CHARMM36 and a user-defined one)
omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", "./specialresidue.xml"], pdbfile=lastpdbfile, periodic=True,
            platform='CPU', numcores=numcores, autoconstraints=None, rigidwater=False)

#QM theory: r2SCAN-3c DFT-composite method using ORCA
orca = ORCATheory(orcasimpleinput="! r2SCAN-3c tightscf", numcores=1)
#QM/MM theory
qmmm = QMMMTheory(qm_theory=orca, mm_theory=omm, fragment=fragment, qm_charge=-1, qm_mult=6,
        embedding="Elstat", qmatoms=qmatoms, printlevel=1, unusualboundary=True)

#Simple single-point calculation
Singlepoint(theory=qmmm, fragment=fragment)
