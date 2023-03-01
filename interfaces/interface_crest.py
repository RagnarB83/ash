import os
import time
import sys
import shutil
import subprocess as sp

from ash.modules.module_coords import split_multimolxyzfile
from ash.functions.functions_general import ashexit, BC, int_ranges, listdiff, print_line_with_subheader1,print_time_rel, pygrep
from ash.modules.module_coords import check_charge_mult, Fragment
import ash.settings_ash

#Very simple crest interface
def call_crest(fragment=None, xtbmethod=None, crestdir=None, charge=None, mult=None, solvent=None, energywindow=6, numcores=1, 
               constrained_atoms=None, forceconstant_constraint=0.5):
    print_line_with_subheader1("call_crest")
    module_init_time=time.time()
    if crestdir == None:
        print(BC.WARNING, "No crestdir argument passed to call_crest. Attempting to find crestdir variable inside settings_ash", BC.END)
        try:
            print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
            crestdir=ash.settings_ash.settings_dict["crestdir"]
        except:
            print(BC.WARNING,"Found no crestdir variable in settings_ash module either.",BC.END)
            try:
                crestdir = os.path.dirname(shutil.which('crest'))
                print(BC.OKGREEN,"Found crest in path. Setting crestdir to:", crestdir, BC.END)
            except:
                print("Found no crest executable in path. Exiting... ")
                ashexit()

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, "QM", fragment, "call_crest", theory=None)

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





#Very simple crest interface for entropy calculations
def call_crest_entropy(fragment=None, crestdir=None, charge=None, mult=None, numcores=1):
    print_line_with_subheader1("call_crest")
    module_init_time=time.time()
    if crestdir == None:
        print(BC.WARNING, "No crestdir argument passed to call_crest. Attempting to find crestdir variable inside settings_ash", BC.END)
        try:
            print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
            crestdir=ash.settings_ash.settings_dict["crestdir"]
        except:
            print(BC.WARNING,"Found no crestdir variable in settings_ash module either.",BC.END)
            try:
                crestdir = os.path.dirname(shutil.which('crest'))
                print(BC.OKGREEN,"Found crest in path. Setting crestdir to:", crestdir, BC.END)
            except:
                print("Found no crest executable in path. Exiting... ")
                ashexit()

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, "QM", fragment, "call_crest", theory=None)

    try:
        shutil.rmtree('crest-calc')
    except:
        pass
    os.mkdir('crest-calc')
    os.chdir('crest-calc')


    #Create XYZ file from fragment (for generality)
    fragment.write_xyzfile(xyzfilename="initial.xyz")


    print("Running crest with entropy option")
    outputfile="crest-ash.out"
    logfile = open(outputfile, 'w')
    #Using POpen and piping like this we can write to stdout and a logfile
    process = sp.Popen([crestdir + '/crest', 'initial.xyz', '--entropy', '-T', str(numcores), '-chrg', str(charge), '-uhf', str(mult-1)], 
        stdout=sp.PIPE, stderr=sp.STDOUT, universal_newlines=True)
    for line in process.stdout:
        sys.stdout.write(line)
        logfile.write(line)
    process.wait()
    logfile.close()
    #TODO: Grab stuff from output
    Sconf = pygrep("   Sconf   =", outputfile)[-1]
    dSrrho = pygrep(" + δSrrho  =", outputfile)[-1]
    Stot = pygrep(" = S(total)  =", outputfile)[-1]
    H_T_0_corr = pygrep("   H(T)-H(0) =", outputfile)[-1]
    G_tot = pygrep(" = G(total)  =", outputfile)[-2]
    entropydict={'Sconf':Sconf,'dSrrho':dSrrho,'Stot':Stot,'H_T_0_corr':H_T_0_corr,'G_tot':G_tot}

    print("Stot:", Stot)
    print("entropydict:", entropydict)
    os.chdir('..')
    print_time_rel(module_init_time, modulename='crest run', moduleindex=0)




    return entropydict






#Grabbing crest conformers. Goes inside rest-calc dir and finds file called crest_conformers.xyz
#Creating ASH fragments for each conformer
def get_crest_conformers(crest_calcdir='crest-calc',conf_file="crest_conformers.xyz", charge=None, mult=None):
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

    for (els,cs,eny) in zip(all_elems,all_coords,list_xtb_energies):
        conf = Fragment(elems=els, coords=cs, charge=charge, mult=mult)
        list_conformers.append(conf)
        conf.energy=eny

    os.chdir('..')
    print("")
    return list_conformers, list_xtb_energies
