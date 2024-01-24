from ash import *
import os
import glob

#Script for calculating selected localized orbitals for each image of an NEB minimum energy path.

#The file containing multiple fragments. Here NEB MEP path XYZ file
knarrfile="knarr_MEP.xyz"

# Atoms for which we want to grab localized orbitals
#Used as input to orblocfind which parses localized orbital output from ORCA
ATOMS = ["52H", "54H"]

#Grab molecules from multi-fragment XYZ-files, Returns list of fragments
frags = get_molecules_from_trajectory(knarrfile,writexyz=True)

#Defining theory level
blocks="""
%loc
locmet iaoibo
end
"""
theory = ORCATheory(orcasimpleinput="! r2scan-3c tightscf", orcablocks=blocks)

#Looping over fragments created and running ORCA localization and creating Cube files 
for i,frag in enumerate(frags):
    j=i+1 #Here using 1-based indexing
    print("Now doing fragment number", j)
    Singlepoint(theory=theory, fragment=frag, charge=0, mult=1)
    os.rename("orca.out", f"orcaloccalc{j}.out") #Renaming output file

    # Grab MO-numbers
    MONUMBERS_alpha, MONUMBERS_beta = orblocfind("orcaloccalc1.out", atomindices=ATOMS)

    #Create cubefiles for selected MO-numbers
    for MONUMBER in MONUMBERS_alpha:
        run_orca_plot("orca.loc", "mo", gridvalue=80, mo_operator=0, mo_number=MONUMBER)
        os.rename(f"orca.mo{MONUMBER}a.cube", f"orcaloccalc{j}_mo{MONUMBER}.cube") #Renaming cube file

    os.rename("orca.loc", f"orcaloccalc{j}.loc") #Renaming loc file 