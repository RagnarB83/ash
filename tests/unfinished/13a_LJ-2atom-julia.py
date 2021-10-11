from ash import *
import sys

#Define global system settings ( scale, tol and conndepth keywords for connectivity)

waterfrag = Fragment(xyzfile="/Users/bjornsson/ownCloud/ASH-tests/testsuite/O-O-dimer.xyz")
print("Numatoms in frag:", waterfrag.numatoms)
#Creating atom-types list
atomtypes=['OT', 'OT']

#Creating charges list
atomcharges=[1.0,2.0]
#Read from file
MM_forcefield=MMforcefield_read('/Users/bjornsson/ownCloud/ASH-tests/testsuite/forcefield-OO.ff')
print(MM_forcefield.items())
for atomtype,property in MM_forcefield.items():
    print("Atomtype:", atomtype)
    print("Charge:", property.atomcharge)
    print("LJParameters:", property.LJparameters)
    blankline()


#PointchargeMMtheory = Theory( atom_types=atomtypes force_field=MMdefinition)
MyMMtheory = NonBondedTheory(charges = atomcharges, atomtypes=atomtypes, forcefield=MM_forcefield, LJcombrule='mixed_geoepsilon', codeversion='py')
timestampA=time.time()
epy,gradpy = Singlepoint(theory=MyMMtheory,fragment=waterfrag)
print_time_rel(timestampA,modulename='py')

timestampA=time.time()
MyMMtheory = NonBondedTheory(charges = atomcharges, atomtypes=atomtypes, forcefield=MM_forcefield, LJcombrule='mixed_geoepsilon', codeversion='julia')
ejulia,gradjulia = Singlepoint(theory=MyMMtheory,fragment=waterfrag)
print_time_rel(timestampA,modulename='julia')

print("")
print("MM Energy (py)", epy)
print("MM Energy (julia)", ejulia)
print("")
print("MM Grad (py)", gradpy)
print("MM Grad (julia)", gradjulia)
