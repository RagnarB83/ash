from ash import *

numcores=8

CO = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

#MP2-unrelaxed
blocks_ur="""
%mp2
density unrelaxed
natorbs true
end
"""
MP2_ur = ORCATheory(orcasimpleinput="! MP2 cc-pVDZ verytightscf", orcablocks=blocks_ur, numcores=numcores, filename="MP2-unrelax")

#MP2-relaxed
blocks_r="""
%mp2
density relaxed
natorbs true
end
"""
MP2_r = ORCATheory(orcasimpleinput="! MP2 cc-pVDZ verytightscf", orcablocks=blocks_r, numcores=numcores, filename="MP2-relax")

#OOMP2
blocks_r="""
%mp2
density relaxed
natorbs true
end
"""
OOMP2 = ORCATheory(orcasimpleinput="! OO-RI-MP2 cc-pVDZ autoaux verytightscf", orcablocks=blocks_r, numcores=numcores, filename="OOMP2")

#MP2 unrelaxed
Singlepoint(theory=MP2_ur, fragment=CO)
mfile = make_molden_file_ORCA(f"{MP2_ur.filename}.mp2nat")
#MP2 relaxed
Singlepoint(theory=MP2_r, fragment=CO)
mfile = make_molden_file_ORCA(f"{MP2_r.filename}.mp2nat")
#OO-MP2
Singlepoint(theory=OOMP2, fragment=CO)
mfile = make_molden_file_ORCA(f"{OOMP2.filename}.mp2nat")
