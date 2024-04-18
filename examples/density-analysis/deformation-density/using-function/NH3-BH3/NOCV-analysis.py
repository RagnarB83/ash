from ash import *

#########################################
# NOCV and Deformation density
##########################################
#Use if you want to calculate the deformation density
#for a molecule (AB) that you split into fragments A and B
#And get full NOCV analysis

#Use geometry for molecule AB and split the file manually into xyz-files A and B
fragment_AB=Fragment(xyzfile="nh3_bh3.xyz", charge=0, mult=1,label='NH3-BH3')
fragment_A=Fragment(xyzfile="nh3.xyz", charge=0, mult=1,label='NH3')
fragment_B=Fragment(xyzfile="bh3.xyz", charge=0, mult=1,label='BH3')

calc = ORCATheory(orcasimpleinput="! RKS BP86 def2-SVP tightscf", orcablocks="")

# Call NOCV_density_ORCA
NOCV_density_ORCA(fragment_AB=fragment_AB, fragment_A=fragment_A, fragment_B=fragment_B, theory=calc, NOCV=True)
