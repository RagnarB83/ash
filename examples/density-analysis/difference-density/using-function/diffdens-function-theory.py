from ash import *

#Examples on how to use the difference_density_ORCA function for different cases

#Example B: Fragments the same but theories differ
HF_neut=Fragment(xyzfile="hf.xyz", charge=0, mult=1)

calcA = ORCATheory(orcasimpleinput="! UKS BP86 def2-SVP", orcablocks="")
calcB = ORCATheory(orcasimpleinput="! UKS HF def2-SVP", orcablocks="")


difference_density_ORCA(fragment_A=HF_neut, fragment_B=HF_neut, theory_A=calcA, theory_B=calcB, griddensity=80)

