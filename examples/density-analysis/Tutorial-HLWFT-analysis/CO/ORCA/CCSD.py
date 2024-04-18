from ash import *

numcores=4

CO = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

#CCSD-lineared
blocks_lin="""
%mdci
density linearized
natorbs true
end
"""
CCSD_lin = ORCATheory(orcasimpleinput="! CCSD cc-pVDZ verytightscf", orcablocks=blocks_lin, numcores=numcores, filename="CCSD-lin")

#CCSD-unrelaxed
blocks_ur="""
%mdci
density unrelaxed
natorbs true
end
"""
CCSD_ur = ORCATheory(orcasimpleinput="! CCSD cc-pVDZ verytightscf", orcablocks=blocks_ur, numcores=numcores, filename="CCSD-unrelax")

#CCSD-orbopt
blocks_r="""
%mdci
density orbopt
natorbs true
end
"""
CCSD_orbopt = ORCATheory(orcasimpleinput="! MP2 cc-pVDZ verytightscf", orcablocks=blocks_r, numcores=numcores, filename="CCSD-orbopt")

#CCSD linearized
Singlepoint(theory=CCSD_lin, fragment=CO)
mfile = make_molden_file_ORCA(f"{CCSD_lin.filename}.mdci.nat")
#CCSD unrelaxed
Singlepoint(theory=CCSD_ur, fragment=CO)
mfile = make_molden_file_ORCA(f"{CCSD_ur.filename}.mdci.nat")
#CCSD orbopt
Singlepoint(theory=CCSD_orbopt, fragment=CO)
mfile = make_molden_file_ORCA(f"{CCSD_orbopt.filename}.mdci.nat")
