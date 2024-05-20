import subprocess as sp
import os
import shutil

from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader,listdiff
import ash.settings_ash

# Basic interface to DRACO: scaling of solvation radii based on geometry and charges
# https://github.com/grimme-lab/DRACO

# This interface simply calls DRACO on an ASH-Fragment or XYZ-file and chosen solvent and returns the scaled radii

# Can be used (with modifications) as input radii in ORCA and other codes

def get_draco_radii(fragment=None, xyzfile=None, charge=None, dracodir=None, radii_type='cpcm', solvent='water',
                    chargemodel='ceh'):

    #Early exits
    if fragment is None and xyzfile is None:
        print(BC.FAIL, "No fragment or xyzfile provided. Exiting.", BC.END)
        ashexit()

    if fragment != None:
        print("Using ASH fragment. Writing XYZ-file to disk")
        xyzfile = fragment.write_xyzfile(xyzfilename="fragment.xyz")
        charge=fragment.charge #Setting charge from fragment
    #Checking if charge was set (by user or from fragment)
    if charge is None:
        print("Error: you must provide a charge for the molecule")
        ashexit()

    dracodir=check_DRACO_location(dracodir)
    print("DRACO directory:", dracodir)
    print("\nXYZ-file:", xyzfile)
    print("Radii type:", radii_type)
    print("Solvent:", solvent)

    print("Running draco. Will write output to draco.out")
    with open('draco.out', 'w') as ofile:
        args_list = [dracodir+'/draco', xyzfile, '--radii', radii_type, '--solvent',
                     solvent,'--charge', str(charge), '--chrgmodel', chargemodel]
        print("args_list:", args_list)
        sp.run(args_list, stdout=ofile)

    #Grab radii from draco.out
    radii = grab_radii('draco.out')
    print("Found Draco radii:", radii)
    return radii

def grab_radii(outfile):
    grab=False
    radii=[]
    with open(outfile) as o:
        for line in o:
            if grab is True:
                if len(line.split()) == 4:
                    radii.append(line.split()[2])
            if 'Identifier  Partial Charge' in line:
                grab=True
    return radii

def check_DRACO_location(dracodir):
    if dracodir != None:
        finaldracodir = dracodir
        print(BC.OKGREEN,f"Using dracodir path provided: {finaldracodir}", BC.END)
    else:
        print(BC.WARNING, "No dracodir argument passed. Attempting to find dracodir variable in ASH settings file (~/ash_user_settings.ini)", BC.END)
        try:
            finaldracodir=ash.settings_ash.settings_dict["dracodir"]
            print(BC.OKGREEN,"Using dracodir path provided from ASH settings file (~/ash_user_settings.ini): ", finaldracodir, BC.END)
        except KeyError:
            print(BC.WARNING,"Found no dracodir variable in ASH settings file either.",BC.END)
            print(BC.WARNING,"Checking for draco in PATH environment variable.",BC.END)
            try:
                finaldracodir = os.path.dirname(shutil.which('draco'))
                print(BC.OKGREEN,"Found draco binary in PATH. Using the following directory:", finaldracodir, BC.END)
            except TypeError:
                print(BC.FAIL,"Found no draco binary in PATH environment variable either. Giving up.", BC.END)
                ashexit()
    return finaldracodir
