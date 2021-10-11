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

#xTB
xtbdir="/opt/xtb-6.4.0/bin"
xtbmethod='GFN2'

#QM and MM objects
XTBQMpart = xTBTheory(xtbdir=xtbdir, charge=0, mult=1, xtbmethod=xtbmethod)
MMpart = NonBondedTheory(charges = atomcharges, atomtypes=atomtypes, forcefield=MM_forcefield, LJcombrule='geometric')

#Create QM/MM theory object. fragment always defined with it
QMMMobject = QMMMTheory(fragment=H2O_MeOH, qm_theory=XTBQMpart,
                                       mm_theory=MMpart, qmatoms=qmatoms, charges=atomcharges,
                                       embedding='Elstat')

#sys.exit(1)
print("Some issue here in xtB/MM geometry optimization. Need to check if pcgradient is faulty")
#Basic Cartesian optimization with KNARR-LBFGS
#with Frozen MM atoms, H2O: 0,1,2
SimpleOpt(theory=QMMMobject, optimizer='KNARR-LBFGS', frozen_atoms=[0,1,2])

sys.exit(0)
