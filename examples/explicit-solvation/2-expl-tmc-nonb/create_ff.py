from ash import *

#Create a nonbonded FF for molecule
frag = Fragment(xyzfile="fecl4.xyz", charge=-1, mult=6)
#Defining QM-theory to be used for charge calculation
orca_theory = ORCATheory(orcasimpleinput="! r2SCAN-3c tightscf")
#
write_nonbonded_FF_for_ligand(fragment=frag, resname="LIG", theory=orca_theory,
        coulomb14scale=1.0, lj14scale=1.0, charge_model="CM5_ORCA")
