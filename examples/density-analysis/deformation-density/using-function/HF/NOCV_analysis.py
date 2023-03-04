from ash import *

#########################################
# Deformation density using function
##########################################
#Use this script if you want to calculate the deformation density
#for a molecule (AB) that you split into fragments A and B
#and do NOCV analysis

#Use geometry for molecule AB and split the file manually into xyz-files A and B
fragment_AB=Fragment(xyzfile="hf.xyz", charge=0, mult=1)
fragment_A=Fragment(xyzfile="h.xyz", charge=0, mult=2)
fragment_B=Fragment(xyzfile="f.xyz", charge=0, mult=2)

calc = ORCATheory(orcasimpleinput="! UKS BP86 def2-SVP", orcablocks="")

# Call NOCV_density_ORCA but with NOCV=False, only deformation density calculated
NOCV_density_ORCA(fragment_AB=fragment_AB, fragment_A=fragment_A, fragment_B=fragment_B, theory=calc, NOCV=True)

