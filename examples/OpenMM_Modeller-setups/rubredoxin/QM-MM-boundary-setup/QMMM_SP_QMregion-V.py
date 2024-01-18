from ash import *

#QM/MM single-point calculation to demonstrate different QM-regions
#
#QM-I: Fe and 4 S atoms only: 5 QM-atoms + 4 linkatoms
#QM-II: Fe and 4 SCH2 group only: 17 QM-atoms + 4 linkatoms
#QM-III (bad): QM-II + C_alpha and H_alpha of Cys-6 backbone. ASH will not allow this
#QM-IV: QM-II + CH(alpha) and NH of Cys-6 backbone + CO of Val-5 backbone
#QM-V: QM-IV + CH-NH of Thr-7 backbone + part of Thr-7 sidechain

#Define number of cores variable
numcores=1

#QM-V:
qmatoms= [93,94,95,96,133,134,135,136,564,565,566,567,604,605,606,607,755,89,90,87,88,75,76,91,92,97,98,99,100,103,104,105,106]

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
        embedding="Elstat", qmatoms=qmatoms, printlevel=1)

#Simple single-point calculation
Singlepoint(theory=qmmm, fragment=fragment)
