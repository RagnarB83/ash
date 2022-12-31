from ash import *

#H2O...MeOH fragment defined. Reading XYZ file
H2O_MeOH = Fragment(xyzfile="h2o_MeOH.xyz")

# Specifying the QM atoms (3-8) by atom indices (MeOH). The other atoms (0,1,2) is the H2O and MM.
#IMPORTANT: atom indices begin at 0.
qmatoms=[3,4,5,6,7,8]

# Charge definitions for whole fragment. Charges for the QM atoms are not important (ASH will always set QM atoms to zero)
atomcharges=[-0.8, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#Defining atomtypes for whole system
atomtypes=['OT','HT','HT','CX','HX', 'HX', 'HX', 'OT', 'HT']

#Read forcefield (here containing LJ-part only) from file
MM_forcefield=MMforcefield_read('MeOH_H2O-sigma.ff')

#QM and MM objects
ORCAQMpart = ORCATheory(orcasimpleinput="!BP86 def2-SVP def2/J tightscf", orcablocks="")
MMpart = NonBondedTheory(charges = atomcharges, atomtypes=atomtypes, forcefield=MM_forcefield, 
    LJcombrule='geometric', codeversion="py")
QMMMobject = QMMMTheory(fragment=H2O_MeOH, qm_theory=ORCAQMpart, mm_theory=MMpart, qmatoms=qmatoms,
                        charges=atomcharges, embedding='Elstat')

#Single-point energy calculation of QM/MM object
result = Singlepoint(theory=QMMMobject, fragment=H2O_MeOH, charge=0, mult=1)

print("Single-point QM/MM energy:", result.energy)

#Geometry optimization of QM/MM object (this may not converge)
result2 = Optimizer(fragment=H2O_MeOH, theory=QMMMobject, coordsystem='tric', ActiveRegion=True, actatoms=[3,4,5,6,7,8], charge=0, mult=1)
print("Optimized QM/MM energy:", result2.energy)
