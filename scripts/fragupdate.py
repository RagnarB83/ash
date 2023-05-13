#!/usr/bin/env python3
"""
Stand-alone script for updating ASH fragment using XYZ information. Companion script to fragedit.py

Usage:
python3 fragupdate.py fragfile.xyz (assumes fragment.xyz and qmatoms file present)
python3 fragupdate.py fragfile.xyz filewithindices ((assumes fragment.xyz )

Reads in ASH fragment file and qmatoms and XYZ file and updates ASH fragment

"""


import sys
#from functions_general import read_intlist_from_file
#from functions_coords import read_xyzfile

# Give difference of two lists, sorted. List1: Bigger list
def listdiff(list1, list2):
    diff = (list(set(list1) - set(list2)))
    diff.sort()
    return diff

#Read lines of file by slurping.
def readlinesfile(filename):
  try:
    f=open(filename)
    out=f.readlines()
    f.close()
  except IOError:
    print('File %s does not exist!' % (filename))
    exit(12)
  return out

#Is variable an integer
def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

#Read list of integers from file. Output list of integers. Ignores blanklines, return chars, non-int characters
#offset option: shifts integers by a value (e.g. 1 or -1)
def read_intlist_from_file(file,offset=0):
    list=[]
    lines=readlinesfile(file)
    for line in lines:
        for l in line.split():
            #Removing non-numeric part
            l = ''.join(i for i in l if i.isdigit())
            if isint(l):
                list.append(int(l)+offset)
    list.sort()
    return list

#Read XYZ file
def read_xyzfile(filename):
    #Will accept atom-numbers as well as symbols
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
            'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
            'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
            'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
            'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    print("Reading coordinates from XYZfile {} ".format(filename))
    coords=[]
    elems=[]
    with open(filename) as f:
        for count,line in enumerate(f):
            if count == 0:
                numatoms=int(line.split()[0])
            if count > 1:
                if len(line.strip()) > 0:
                    if isint(line.split()[0]) is True:
                        elems.append(elements[int(line.split()[0])-1])
                    else:
                        elems.append(line.split()[0])
                    coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
    assert len(coords) == numatoms, "Number of coordinates does not match header line"
    assert len(coords) == len(elems), "Number of coordinates does not match elements."
    return elems,coords



try:
    fragfile=sys.argv[1]
except:
    print("Please provide an XYZ-file as argument")
    exit(1)
#Try to process a qmatoms file if provided
try:
    qmatoms_file = sys.argv[2]
    qmatoms = read_intlist_from_file(qmatoms_file)
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

#Read ASH fragfile (XYZ-file)
with open(fragfile) as file:
    for line in file:
        fragfile_lines.append(line)


#Write modified ASH XYZ-file
print("xyz_elems:", xyz_elems)
print("xyz_coords:", xyz_coords)
atindex=0
with open(fragfile,'w') as newfile:
    for i,line in enumerate(fragfile_lines):
        if i-2 in qmatoms:
            print("i-2:", i-2)
            newel = xyz_elems.pop(0)
            newcoords=xyz_coords.pop(0)
            newline = f"{newel} {newcoords[0]} {newcoords[1]} {newcoords[2]}"
            print("old line:", line)
            print("newline:", newline)
            newfile.write(newline)
        else:
            newfile.write(line)

print("xyz_elems:", xyz_elems)
print("xyz_coords:", xyz_coords)
# coordline=False
# with open(fragfile, 'w') as newfile:
#     for line in fragfile_lines:
#         if '=====' in line:
#             coordline=False
#         if coordline==True:
#             if int(line.split()[0]) in qmatoms:
#                 #print("int(line.split()[0]):", int(line.split()[0]))
#                 #print("line:", line)
#                 at=int(line.split()[0])
#                 el=line.split()[1]
#                 charge=float(line.split()[5])
#                 label=line.split()[6]
#                 atomtype=line.split()[7]
#                 newcoords=xyz_coords.pop(0)
#                 #print("newcoords:", newcoords)
#                 line = "{:>6} {:>6}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f} {:12} {:>21}\n".format(at, el, newcoords[0],
#                                                                                                     newcoords[1], newcoords[2],
#                                                                                                     charge, label, atomtype)
#         if '-----------------------' in line:
#             coordline=True

#         newfile.write(line)
print(f"Updated file {fragfile} with coordinates from file {xyzfile}")