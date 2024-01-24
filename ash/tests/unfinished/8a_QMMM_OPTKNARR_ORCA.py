from ash import *
import sys
#Define global system settings ( scale, tol and conndepth keywords for connectivity)

#H2O...MeOH fragment
H2O_MeOH = Fragment(xyzfile="/Users/bjornsson/ownCloud/ASH-tests/testsuite/h2o_MeOH.xyz")

# MeOH qm atoms. Rest: 0,1,2 is H2O and MM
qmatoms=[3,4,5,6,7,8]
# Charge definitions for whole fragment.
atomcharges=[-0.8, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#MM info
atomtypes=['OT','HT','HT','CX','HX', 'HX', 'HX', 'OT', 'HT']
#Read from file
MM_forcefield=MMforcefield_read('/Users/bjornsson/ownCloud/ASH-tests/testsuite/MeOH_H2O-sigma.ff')

#ORCA
orcadir='/opt/orca_4.2.1'
orcasimpleinput="! PBE def2-SVP Grid5 Finalgrid6 tightscf"
orcablocks="%scf maxiter 200 end"

#QM and MM objects
ORCAQMpart = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)
MMpart = NonBondedTheory(charges = atomcharges, atomtypes=atomtypes, forcefield=MM_forcefield, LJcombrule='geometric')

#Create QM/MM theory object. fragment always defined with it
QMMM_SP = QMMMTheory(fragment=H2O_MeOH, qm_theory=ORCAQMpart,
                                       mm_theory=MMpart, qmatoms=qmatoms, charges=atomcharges,
                                       embedding='Elstat')

#Simple Energy SP calc
Singlepoint(theory=QMMM_SP,fragment=H2O_MeOH)


sys.exit(0)
