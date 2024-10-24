from ash import *

# Defining fragment
mol = Fragment(xyzfile="3fgaba.xyz", charge=0, mult=1)

# OPTION 1: Full FF (light elements only)
# Parameterize small molecule using OpenFF (only for simple, usually organics-only molecules)
small_molecule_parameterizer(xyzfile="3fgaba.xyz", forcefield_option="OpenFF", charge=0)
# Creates file: openff_LIG.xml

# OPTION 2: Nonbonded FF
# Defining QM-theory to be used for charge calculation
theory = ORCATheory(orcasimpleinput="! r2SCAN-3c tightscf")
# Calling write_nonbonded_FF_for_ligand to create a simple nonbonded FF
write_nonbonded_FF_for_ligand(fragment=mol, resname="LIG", theory=theory,
   charge_model="CM5_ORCA", LJ_model="UFF")
# Creates file : LIG.xml