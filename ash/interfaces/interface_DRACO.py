import subprocess as sp
import os
import shutil

from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader,listdiff
import ash.settings_ash

# Basic interface to DRACO: scaling of solvation radii based on geometry and charges
# https://github.com/grimme-lab/DRACO

# This interface simply calls DRACO on an XYZ-file and chosen solvent and returns the scaled radii
# TODO: Add qmodeL EEQ vs CEH option

# Can be used (with modifications) as input radii in ORCA and other codes


def get_draco_radii(fragment=None, xyzfile=None, dracodir=None, radii_type='cpcm', solvent=None):

    if fragment == None and xyzfile == None:
        print(BC.FAIL, "No fragment or xyzfile provided. Exiting.", BC.END)
        ashexit()
    if fragment != None:
        print("Using ASH fragment. Writing XYZ-file to disk")
        xyzfile = fragment.write_xyzfile(xyzfilename="fragment.xyz")

    dracodir=check_DRACO_location(dracodir)
    print("DRACO directory:", dracodir)
    print("\nXYZ-file:", xyzfile)
    print("Radii type:", radii_type)
    print("Solvent:", solvent)

    #draco h2o.xyz --solvent water
    with open('draco.out', 'w') as ofile:
        sp.run([dracodir+'/draco', xyzfile, '--radii', radii_type, '--solvent', solvent], stdin=input, stdout=ofile)

    #Grab radii from draco.out
    radii = grab_radii('draco.out')

    return radii

def grab_radii(outfile):
    radii=[]
    with open(outfile) as o:
        for line in o:
            if len(line.split()) == 4:
                if 'Identifier' not in line:
                    radii.append(line.split()[2])
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
