#!/bin/env python3
"""
fragedit.py


"""
import sys
#from functions_general import read_intlist_from_file
#from functions_coords import write_xyzfile


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

#Write XYZfile provided list of elements and list of list of coords and filename
def write_xyzfile(elems,coords,name,printlevel=2):
    with open(name+'.xyz', 'w') as ofile:
        ofile.write(str(len(elems))+'\n')
        ofile.write("title"+'\n')
        for el,c in zip(elems,coords):
            line="{:4} {:16.12f} {:16.12f} {:16.12f}".format(el,c[0], c[1], c[2])
            ofile.write(line+'\n')
    if printlevel >= 2:
        print("Wrote XYZ file:", name+'.xyz')
        
#Standalone fragment-editing script for Ash

#Reads in Ash fragment file and qmatoms and output XYZ coordinate file that can be visualized in e.g. Chemcraft and edited

#Fragfile is always first argument
try:
    fragfile=sys.argv[1]
except:
    print("Please provide an Ash fragment file as argument")
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

print("qmatoms list: ", qmatoms)
print("Grabbing QM-region coordinates...")
coordline=False
elems=[]
coords=[]
with open(fragfile) as file:
    for line in file:
        if '=====' in line:
            coordline=False
        if coordline==True:
            if int(line.split()[0]) in qmatoms:
                el=line.split()[1]
                c_x=float(line.split()[2]);c_y=float(line.split()[3]);c_z=float(line.split()[4])
                elems.append(el)
                coords.append([c_x, c_y,c_z])
        if '-----------------------' in line:
            coordline=True

write_xyzfile(elems,coords,"fragment")
