import os
import time
import shutil

from modules.module_coords import split_multimolxyzfile
from functions.functions_general import ashexit, BC, int_ranges, listdiff, print_line_with_subheader1,print_time_rel
import subprocess as sp
import ash


#Very simple crest interface
def call_crest(fragment=None, xtbmethod=None, crestdir=None, charge=None, mult=None, solvent=None, energywindow=6, numcores=1, 
               constrained_atoms=None, forceconstant_constraint=0.5):
    print_line_with_subheader1("call_crest")
    module_init_time=time.time()
    if crestdir == None:
        print(BC.WARNING, "No crestdir argument passed to call_crest. Attempting to find crestdir variable inside settings_ash", BC.END)
        try:
            print("settings_ash.settings_dict:", settings_ash.settings_dict)
            crestdir=settings_ash.settings_dict["crestdir"]
        except:
            print(BC.WARNING,"Found no crestdir variable in settings_ash module either.",BC.END)
            try:
                crestdir = os.path.dirname(shutil.which('crest'))
                print(BC.OKGREEN,"Found crest in path. Setting crestdir to:", crestdir, BC.END)
            except:
                print("Found no crest executable in path. Exiting... ")
                ashexit()

    #Use charge/mult from frag if charge/mult keywords not set
    if charge == None and mult == None:
        print(BC.WARNING,"Warning: No charge/mult was defined for call_crest. Checking fragment.",BC.END)
        if fragment.charge != None and fragment.mult != None:
            print(BC.WARNING,"Fragment contains charge/mult information: Charge: {} Mult: {} Using this instead".format(fragment.charge,fragment.mult), BC.END)
            print(BC.WARNING,"Make sure this is what you want!", BC.END)
            charge=fragment.charge; mult=fragment.mult
            theory_chargemult_change=True
        else:
            print(BC.FAIL,"No charge/mult information present in fragment either. Exiting.",BC.END)
            ashexit()


    try:
        shutil.rmtree('crest-calc')
    except:
        pass
    os.mkdir('crest-calc')
    os.chdir('crest-calc')

    if constrained_atoms != None:
        allatoms=range(0,fragment.numatoms)
        unconstrained=listdiff(allatoms,constrained_atoms)
                        
        constrained_crest=[i+1 for i in constrained_atoms]
        unconstrained_crest=[j+1 for j in unconstrained]
        
        #Get ranges. List of tuples
        constrained_ranges=int_ranges(constrained_crest)
        unconstrained_ranges=int_ranges(unconstrained_crest)
        
        
        print("Creating .xcontrol file for constraints")
        with open(".xcontrol","w") as constrainfile:
            constrainfile.write("$constrain\n")
            #constrainfile.write("atoms: {}\n".format(','.join(map(str, constrained_ranges))))
            constrainfile.write("atoms: {}\n".format(constrained_ranges))
            constrainfile.write("force constant={}\n".format(forceconstant_constraint))
            constrainfile.write("$metadyn\n")
            constrainfile.write("atoms: {}\n".format(unconstrained_ranges ))
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
    #GBSA solvation or not
    if solvent is None:
        process = sp.run([crestdir + '/crest', 'initial.xyz', '-T', str(numcores), '-gfn'+str(xtbflag), '-ewin', str(energywindow), '-chrg', str(charge), '-uhf', str(mult-1)])
    else:
        process = sp.run([crestdir + '/crest', 'initial.xyz','-T', str(numcores),  '-gfn' + str(xtbflag), '-ewin', str(energywindow), '-chrg', str(charge),'-gbsa', str(solvent),
             str(charge), '-uhf', str(mult - 1)])


    os.chdir('..')
    print_time_rel(module_init_time, modulename='crest run', moduleindex=0)

    #Get conformers
    list_conformers, list_xtb_energies = get_crest_conformers()


    return list_conformers, list_xtb_energies


#Grabbing crest conformers. Goes inside rest-calc dir and finds file called crest_conformers.xyz
#Creating ASH fragments for each conformer
def get_crest_conformers(crest_calcdir='crest-calc',conf_file="crest_conformers.xyz"):
    print("")
    print("Now finding Crest conformers and creating ASH fragments...")
    os.chdir(crest_calcdir)
    list_conformers=[]
    list_xtb_energies=[]
    all_elems, all_coords, all_titles = split_multimolxyzfile(conf_file,writexyz=True)
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
