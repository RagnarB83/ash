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


#PointchargeMMtheory = Theory( atom_types=atomtypes force_field=MMdefinition)
MyMMtheory = NonBondedTheory(charges = atomcharges, atomtypes=atomtypes, forcefield=MM_forcefield, LJcombrule='geometric', codeversion='py')
timestampA=time.time()
Singlepoint(theory=MyMMtheory,fragment=frag)
print_time_rel(timestampA,modulename='py')


MyMMtheory = NonBondedTheory(charges = atomcharges, atomtypes=atomtypes, forcefield=MM_forcefield, LJcombrule='geometric', codeversion='julia')
timestampA=time.time()
Singlepoint(theory=MyMMtheory,fragment=frag)
print_time_rel(timestampA,modulename='julia')
