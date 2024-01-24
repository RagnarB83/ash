from ash import *
import os
import glob

numcores=5

xyzfiles_dir="/Users/bjornsson/ownCloud/ASH-tests/testsuite/fragfiles_5"

fragments=[]
for file in glob.glob(xyzfiles_dir+"/*"):
    label=file.split('/')[-1].split('.')[0]
    print(file)
    #frag=Fragment(xyzfile=file, label=label)
    frag=Fragment(xyzfile=file)
    fragments.append(frag)


print(fragments)

#xtbcalc=xTBTheory(charge=0,mult=1, xtbmethod='GFN2-xTB', numcores=1)
orcacalc=ORCATheory(charge=0,mult=1, orcasimpleinput="! HF def2-SVP")
result = Singlepoint_parallel(theories=[orcacalc], fragments=fragments, numcores=numcores)

print("Results object:", result)
print("Final energydict:", result.energies_dict)
