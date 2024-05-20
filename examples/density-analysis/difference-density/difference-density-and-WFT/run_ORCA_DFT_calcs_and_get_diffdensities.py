from ash import *

numcores=8

#File with reference orbitals
reference_orbital_file="hf.gbw"

#Frag
frag = Fragment(databasefile="ethylene.xyz")

#Theories
hf = ORCATheory(orcasimpleinput="! RHF cc-pVTZ tightscf",filename="hf", label="hf", numcores=numcores)
tpss = ORCATheory(orcasimpleinput="! TPSS cc-pVTZ tightscf", filename="tpss", label="tpss", numcores=numcores)
b3lyp = ORCATheory(orcasimpleinput="! B3LYP cc-pVTZ tightscf", filename="b3lyp", label="b3lyp", numcores=numcores)
theories=[hf,tpss,b3lyp]

#Looping over theories
for theory in theories:
    Singlepoint(theory=theory, fragment=frag)

#Diffdens tool will create densities for the GBW files found in the directory
#and do difference density vs. the reference
diffdens_tool(reference_orbfile=reference_orbital_file, dir='.', grid=3)
