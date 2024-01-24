from ash import *
import sys

# Comparison of MM part in a QM/MM job done with different LJCoulomb algorithms
#Comparison done to Chemshell here: /home/bjornsson/grad-test-chemshell/H2o_MEoH


#Define global system settings ( scale, tol and conndepth keywords for connectivity)

frag = Fragment(xyzfile="/Users/bjornsson/ownCloud/ASH-tests/testsuite/h2o_MeOH.xyz")
print("Numatoms in frag:", frag.numatoms)
# Charge definitions for whole fragment.
atomcharges=[-0.8, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
atomtypes=['OT','HT','HT','CX','HX', 'HX', 'HX', 'OT', 'HT']
# MeOH qm atoms. Rest: 0,1,2 is H2O and MM
qmatoms=[3,4,5,6,7,8]

#ORCA
orcadir='/opt/orca_4.2.1'
orcasimpleinput="! PBE def2-SVP Grid5 Finalgrid6 tightscf"
orcablocks="%scf maxiter 200 end"

#Read from file
MM_forcefield=MMforcefield_read('/Users/bjornsson/ownCloud/ASH-tests/testsuite/MeOH_H2O-R0.ff')
print(MM_forcefield.items())
for atomtype,property in MM_forcefield.items():
    print("Atomtype:", atomtype)
    print("Charge:", property.atomcharge)
    print("LJParameters:", property.LJparameters)
    blankline()

print(frag.connectivity)
frag.print_system("h2o_MeOH.ygg")

timestampA=time.time()
#QM and MM. qmatoms list passed to MMpart because of LJ pair-potentials
ORCAQMpart = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)
MMpart = NonBondedTheory(charges = atomcharges, atomtypes=atomtypes, forcefield=MM_forcefield, 
    LJcombrule='geometric', codeversion='py')
QMMM_calc = QMMMTheory(fragment=frag, qm_theory=ORCAQMpart, mm_theory=MMpart, qmatoms=qmatoms, charges=atomcharges,
                                       embedding='Elstat')

Singlepoint(theory=QMMM_calc,fragment=frag)
print_time_rel(timestampA,modulename='py')

timestampA=time.time()

ORCAQMpart = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)
MMpart = NonBondedTheory(charges = atomcharges, atomtypes=atomtypes, forcefield=MM_forcefield,
    LJcombrule='geometric', codeversion='julia')
QMMM_calc = QMMMTheory(fragment=frag, qm_theory=ORCAQMpart, mm_theory=MMpart, qmatoms=qmatoms, charges=atomcharges,
                                       embedding='Elstat')

Singlepoint(theory=QMMM_calc,fragment=frag)
print_time_rel(timestampA,modulename='julia')
