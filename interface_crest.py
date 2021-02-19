import os
from functions_coords import *
import subprocess as sp
from ash import *


#Very simple crest interface
def call_crest(fragment=None, xtbmethod=None, crestdir=None,charge=None, mult=None, solvent=None, energywindow=6, numcores=1, 
               constrained_atoms=None, forceconstant_constraint=0.5):

    os.mkdir('crest-calc')
    os.chdir('crest-calc')

    if constrained_atoms != None:
        allatoms=range(0,fragment.numatoms)
        unconstrained=listdiff(allatoms,constrained_atoms)
        constrained_crest=[i+1 for i in constrained_atoms]
        unconstrained_crest=[j+1 for j in unconstrained]
        print("Creating .xcontrol file for constraints")
        with open(".xcontrol","w") as constrainfile:
            constrainfile.write("$constrain\n")
            constrainfile.write("atoms: {}\n".format(','.join(map(str, constrained_crest))))
            constrainfile.write("force constant={}\n".format(forceconstant_constraint))
            constrainfile.write("$metadyn\n")
            constrainfile.write("atoms: {}\n".format(','.join(map(str, unconstrained_crest ))))
            constrainfile.write("$end\n")

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
        process = sp.run([crestdir + '/crest', 'initial.xyz', '-T', str(numcores), '-gfn'+str(xtbflag), '-ewin', str(energywindow), '-chrg', str(charge), '-uhf', str(mult-1)])
    else:
        process = sp.run([crestdir + '/crest', 'initial.xyz','-T', str(numcores),  '-gfn' + str(xtbflag), '-ewin', str(energywindow), '-chrg', str(charge),'-gbsa', str(solvent),
             str(charge), '-uhf', str(mult - 1)])

    os.chdir('..')

#Grabbing crest conformers. Assuming inside crest-calc dir and in file called crest_conformers.xyz
#Creating ASH fragments for each conformer
def get_crest_conformers():
    print("")
    print("Now finding Crest conformers and creating ASH fragments...")
    os.chdir('crest-calc')
    list_conformers=[]
    list_xtb_energies=[]
    all_elems, all_coords, all_titles = split_multimolxyzfile("crest_conformers.xyz",writexyz=True)
    print("Found {} Crest conformers".format(len(all_elems)))
    
    #Getting energies from title lines
    for i in all_titles:
        en=float(i)
        list_xtb_energies.append(en)

    for els,cs in zip(all_elems,all_coords):
        conf = ash.Fragment(elems=els, coords=cs)
        list_conformers.append(conf)

    os.chdir('..')
    print("")
    return list_conformers, list_xtb_energies