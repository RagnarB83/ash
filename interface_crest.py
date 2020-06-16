import os
from functions_coords import *
import subprocess as sp
from ash import *

#Very simple crest interface
def call_crest(fragment=None, xtbmethod=None, crestdir=None,charge=None, mult=None, solvent=None, energywindow=6):

    os.mkdir('crest-calc')
    os.chdir('crest-calc')

    #Create XYZ file from fragment (for generality)
    fragment.write_xyzfile(xyzfilename="initial.xyz")
    #Theory level
    if 'GFN2' in xtbmethod.upper():
        xtbflag=2
    elif 'GFN1' in xtbmethod.upper():
        xtbflag=1
    elif 'GFN0' in xtbmethod.upper():
        xtbflag=0
    else:
        print("Using default GFN2-xTB")
        xtbflag=2
    uhf=mult-1
    #GBSA solvation or not
    if solvent is None:
        print("here")
        print(crestdir)
        process = sp.run([crestdir + '/crest', 'initial.xyz', '-gfn'+str(xtbflag), '-ewin', str(energywindow), '-chrg', str(charge), '-uhf', str(mult-1)])
    else:
        process = sp.run([crestdir + '/crest', 'initial.xyz', '-gfn' + str(xtbflag), '-ewin', str(energywindow), '-chrg','-gbsa', str(solvent),
             str(charge), '-uhf', str(mult - 1)])

    os.chdir('..')

#Grabbing crest conformers. Assuming inside crest-calc dir and in file called crest_conformers.xyz
#Creating ASH conformers for each
def get_crest_conformers():
    os.chdir('crest-calc')
    list_conformers=[]
    list_xtb_energies=[]
    all_elems, all_coords, all_titles = split_multimolxyzfile("crest_conformers.xyz",writexyz=True)

    #Getting energies from title lines
    print("all_titles:", all_titles)
    for i in all_titles:
        en=float(i)
        list_xtb_energies.append(en)

    for els,cs in zip(all_elems,all_coords):
        conf = ash.Fragment(elems=els, coords=cs)
        list_conformers.append(conf)

    os.chdir('..')
    return list_conformers, list_xtb_energies