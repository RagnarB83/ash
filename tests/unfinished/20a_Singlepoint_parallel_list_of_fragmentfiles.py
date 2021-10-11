from ash import *
import os
import glob

numcores=5

xyzfiles_dir="/Users/bjornsson/ownCloud/ASH-tests/testsuite/fragfiles_5"

#Creating list of fragmentfiles
fragmentfiles=[]
for file in glob.glob(xyzfiles_dir+"/*"):
    label=file.split('/')[-1].split('.')[0]
    print(file)
    frag=Fragment(xyzfile=file, label=label)
    frag.print_system(filename=label+'.ygg')
    #frag=Fragment(xyzfile=file)
    fragmentfiles.append(label+'.ygg')


print(fragmentfiles)

#xtbcalc=xTBTheory(charge=0,mult=1, xtbmethod='GFN2-xTB', numcores=1)
orcacalc=ORCATheory(charge=0,mult=1, orcasimpleinput="! HF def2-SVP")
energydict = Singlepoint_parallel(theories=[orcacalc], fragmentfiles=fragmentfiles, numcores=numcores)

print("Final energydict:", energydict)
