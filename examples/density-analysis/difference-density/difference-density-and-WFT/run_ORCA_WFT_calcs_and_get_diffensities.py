from ash import *

numcores=8
#File with reference orbitals
reference_orbital_file="hf.gbw"

#Frag
frag = Fragment(databasefile="ethylene.xyz")

#Theories
ccsdblocks="""
%mdci
density unrelaxed
donatorbs true
end
"""

mp2blocks="""
%mp2
density unrelaxed
donatorbs true
end
"""
mp2 = ORCATheory(orcasimpleinput="! MP2 cc-pVTZ tightscf", orcablocks=mp2blocks, filename="mp2", label="mp2", numcores=numcores)
ccsd = ORCATheory(orcasimpleinput="! CCSD cc-pVTZ tightscf", orcablocks=ccsdblocks, filename="ccsd", label="ccsd",numcores=numcores)
theories=[mp2,ccsd]

#Looping over theories
for theory in theories:
    Singlepoint(theory=theory, fragment=frag)
    #Remove GBW file created since we only want the nat-file
    os.remove(f"{theory.filename}.gbw")

#Diffdens tool will create densities for the NAT (and GBW) files found in the directory
#and do difference density vs. the reference
diffdens_tool(reference_orbfile=reference_orbital_file, dir='.', grid=3)
