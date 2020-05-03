import sys
import functions_general
import functions_coords

#Standalone fragment-editing script for Yggdrasill

#Reads in Yggdrasill fragment file and qmatoms and output XYZ coordinate file that can be visualized in e.g. Chemcract and edited

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
    print(qmatoms)

#except:
#    print("Found no file qmatoms on disk")
#    exit(1)

print("qmatoms list:", qmatoms)

coordline=False
elems=[]
coords=[]
with open(fragfile) as file:
    for line in file:
        if coordline==True:
            if int(line.split()[0]) in qmatoms:
                el=line.split()[1]
                c_x=float(line.split()[2]);c_y=float(line.split()[3]);c_z=float(line.split()[4])
                elems.append(el)
                coords.append([c_x, c_y,c_z])
        if '-----------------------' in line:
            coordline=True

write_xyzfile(elems,coords,"fragment.xyz")
