from ash import *

numcores=1

CO = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

#CASSCF(2,2)
casblocks_2_2="""
%casscf
nel 2
norb 2
end
"""
CASSCF_2_2 = ORCATheory(orcasimpleinput="! CASSCF cc-pVDZ verytightscf", orcablocks=casblocks_2_2, numcores=numcores, filename="CAS_2_2")

#CASSCF(6,5)
casblocks_6_5="""
%casscf
nel 6
norb 5
end
"""
CASSCF_6_5 = ORCATheory(orcasimpleinput="! CASSCF cc-pVDZ verytightscf", orcablocks=casblocks_6_5, numcores=numcores, filename="CAS_6_5")

#CASSCF(10,8)
casblocks_10_8="""
%casscf
nel 10
norb 8
end
"""
CASSCF_10_8 = ORCATheory(orcasimpleinput="! CASSCF cc-pVDZ verytightscf", orcablocks=casblocks_10_8, numcores=numcores, filename="CAS_10_8")

#CASSCF 2,2
Singlepoint(theory=CASSCF_2_2, fragment=CO)
mfile = make_molden_file_ORCA(f"{CASSCF_2_2.filename}.gbw")

#CASSCF 6,5
Singlepoint(theory=CASSCF_6_5, fragment=CO)
mfile = make_molden_file_ORCA(f"{CASSCF_6_5.filename}.gbw")


#CASSCF 10,8
Singlepoint(theory=CASSCF_10_8, fragment=CO)
mfile = make_molden_file_ORCA(f"{CASSCF_10_8.filename}.gbw")
