from ash import *

numcores=8

CO = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

#CASSCF(10,8) and MRCI with no truncations
blocks="""
%casscf
nel 10
norb 8
end
%mrci
tsel 0
tpre 0
donatorbs 2
end
"""
MRCI_10_8_tsel_0 = ORCATheory(orcasimpleinput="! MRCI+Q cc-pVDZ verytightscf", orcablocks=blocks, numcores=numcores, filename="MRCI_10_8")

#MRCI+Q 10,8
Singlepoint(theory=MRCI_10_8_tsel_0, fragment=CO)

#Making Molden file from MRCI natural orbital file (note special name)
mfile = make_molden_file_ORCA(f"{MRCI_10_8_tsel_0.filename}.b0_s0.nat")
