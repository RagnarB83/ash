
from ash import *

frag=Fragment(xyzfile="Fe2S2.xyz", charge=2, mult=1)

#Script to get nonbonded model parameters for a ligand
orcatheory=ORCATheory(orcasimpleinput="! UKS r2scan ZORA ZORA-def2-TZVP tightscf CPCM", numcores=1,
    brokensym=True, HSmult=11, atomstoflip=[0])

write_nonbonded_FF_for_ligand(fragment=frag, resname="test", charge=0, mult=1,
    coulomb14scale=1.0, lj14scale=1.0, charge_model="CM5_ORCA", theory=orcatheory, LJ_model="UFF", charmm=True)
