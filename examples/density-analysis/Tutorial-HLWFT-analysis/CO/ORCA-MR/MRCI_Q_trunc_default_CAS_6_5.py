from ash import *

numcores=2

CO = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

#CASSCF(6,5) and MRCI with no truncations
blocks="""
%casscf
nel 6
norb 5
end
%mrci
donatorbs 2
end
"""
MRCI_6_5_tsel_def = ORCATheory(orcasimpleinput="! MRCI+Q cc-pVDZ verytightscf", orcablocks=blocks, numcores=numcores, filename="MRCI_6_5_tsel_def")

#MRCI+Q 6,5 calculation
Singlepoint(theory=MRCI_6_5_tsel_def, fragment=CO)

#Making Molden file from MRCI natural orbital file (note special name)
mfile = make_molden_file_ORCA(f"{MRCI_6_5_tsel_def.filename}.b0_s0.nat")
