from ash import *

#Examples on how to use the difference_density_ORCA function for different cases

#Example A: Fragments differ (vertical redox)
HF_neut=Fragment(xyzfile="hf.xyz", charge=0, mult=1)
HF_ox=Fragment(xyzfile="hf.xyz", charge=1, mult=2)

calc = ORCATheory(orcasimpleinput="! UKS BP86 def2-SVP", orcablocks="")

difference_density_ORCA(fragment_A=HF_neut, fragment_B=HF_ox, theory_A=calc, theory_B=calc, griddensity=80)

