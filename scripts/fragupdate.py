#!/bin/env python3

import sys
from functions_general import read_intlist_from_file
from functions_coords import read_xyzfile

#Update Yggdrasill fragment after fragedit use and XYZ coord modification

#Standalone fragment-update script for Yggdrasill
#Reads in Yggdrasill fragment file and qmatoms and XYZ file and updates Yggdrasill fragment

#Fragfile is always first argument
try:
    fragfile=sys.argv[1]
except:
    print("Please provide an Yggdrasill fragment file as argument")
    exit(1)
#Try to process a qmatoms file if provided
try:
    qmatoms_file = sys.argv[2]
    read_intlist_from_file(qmatoms_file)
except:
    print("No atomlist-file provided as 2nd argument. Attempting to read file named qmatoms from disk")
    qmatoms = read_intlist_from_file("qmatoms")


#sort qmatoms list
qmatoms.sort()
print("qmatoms:", qmatoms)

# Read modified XYZfile
xyzfile="fragment.xyz"
xyz_elems,xyz_coords=read_xyzfile(xyzfile)

coordline=False
elems=[]
coords=[]
fragfile_lines=[]

#Read Yggdrasill fragfile
with open(fragfile) as file:
    for line in file:
        fragfile_lines.append(line)


#Write modified Yggdrasill fragfile
coordline=False
with open(fragfile, 'w') as newfile:
    for line in fragfile_lines:
        if '=====' in line:
            coordline=False
        if coordline==True:
            if int(line.split()[0]) in qmatoms:
                #print("int(line.split()[0]):", int(line.split()[0]))
                #print("line:", line)
                at=int(line.split()[0])
                el=line.split()[1]
                charge=float(line.split()[5])
                label=int(line.split()[6])
                atomtype=line.split()[7]
                newcoords=xyz_coords.pop(0)
                #print("newcoords:", newcoords)
                line = "{:>6} {:>6}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f} {:12d} {:>21}\n".format(at, el, newcoords[0],
                                                                                                    newcoords[1], newcoords[2],
                                                                                                    charge, label, atomtype)
        if '-----------------------' in line:
            coordline=True

        newfile.write(line)
print("Updated file {} with coordinates from file {}".format(fragfile,xyzfile))